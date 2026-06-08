from __future__ import annotations

import asyncio
from typing import Dict, Any

from app.stores.session_store import conv_store
from app.graph.state import GraphState
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GraphNodes:
    """父图节点集合 — 编排 5 个子图 + 3 个协调节点。"""

    def __init__(
        self,
        conversation_graph,
        planner_graph,
        retriever_agent_graph,
        responder_agent_graph,
        critic_agent_graph,
    ) -> None:
        self._conversation = conversation_graph
        self._planner = planner_graph
        self._retriever_agent = retriever_agent_graph
        self._responder_agent = responder_agent_graph
        self._critic_agent = critic_agent_graph

    # ====================================================================
    # ConversationAgent — 入口 Agent
    # ====================================================================

    async def conversation_agent(self, state: GraphState) -> Dict[str, Any]:
        """用户意图理解 + 对话记忆 + 上下文融合。

        ConversationAgent 内部处理：
        1. classify_intent → 闲聊 or 知识问答
        2. 闲聊 → 直接生成答案（Agent 内部完成）
        3. 知识问答 → 加载历史 → 追问识别 → 上下文融合
        """
        question = state.get("question", "")
        session_id = state.get("session_id", "")
        node_log: list = []

        logger.info("[Conversation] session=%s", session_id[:8] or "none")

        sub_result = await self._conversation.ainvoke({
            "question": question,
            "session_id": session_id,
        })

        intent = sub_result.get("intent", "knowledge_query")
        node_log.extend(sub_result.get("agent_log", []))

        result: dict = {
            "original_question": question,
            "chat_history": sub_result.get("chat_history", []),
            "node_log": node_log,
            "max_regenerates": 2,
            "regenerate_count": 0,
            "documents": [],
            "sources": [],
            "retrieval_details": {},
            "critic_verdict": "",
            "critic_issues": [],
            "critic_feedback": "",
        }

        if intent == "direct_answer":
            # Agent 已生成答案，直接返回
            result["final_answer"] = sub_result.get("answer", "")
            result["generation"] = sub_result.get("answer", "")
            result["sources"] = sub_result.get("sources", [])
            result["retrieval_details"] = sub_result.get(
                "retrieval_details", {"doc_count": 0}
            )
        else:
            # 知识问答路径，下游 Agent 使用增强后的问题
            result["question"] = sub_result.get(
                "enriched_question", question
            )

        return result

    # ====================================================================
    # Save Conversation
    # ====================================================================

    async def save_conversation(self, state: GraphState) -> Dict[str, Any]:
        """保存本轮问答到 Redis 对话记忆。"""
        session_id = state.get("session_id", "")
        original_question = state.get("original_question", state["question"])
        final_answer = state.get("final_answer", "")
        node_log = list(state.get("node_log", []))

        if session_id and final_answer:
            conv_store.append(session_id, "user", original_question)
            conv_store.append(session_id, "assistant", final_answer)
            node_log.append("💾 对话已保存到记忆")
            logger.info("[SaveConv] Saved turn to session %s", session_id[:8])

        return {"node_log": node_log}

    # ====================================================================
    # QueryPlanner
    # ====================================================================

    async def query_planner(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        node_log = list(state.get("node_log", []))
        sub_result = await self._planner.ainvoke({"question": question})
        node_log.extend(sub_result.get("agent_log", []))
        return {
            "is_complex": sub_result.get("is_complex", False),
            "sub_queries": sub_result.get("sub_queries", []),
            "node_log": node_log,
        }

    # ====================================================================
    # RetrieverAgent (single)
    # ====================================================================

    async def retriever_agent(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        node_log = list(state.get("node_log", []))
        sub_result = await self._retriever_agent.ainvoke({"query": question})
        node_log.extend(sub_result.get("agent_log", []))
        return {"documents": sub_result.get("documents", []), "node_log": node_log}

    # ====================================================================
    # Multi-Retrieve (parallel)
    # ====================================================================

    async def multi_retrieve(self, state: GraphState) -> Dict[str, Any]:
        sub_queries: list = state.get("sub_queries", [])
        node_log = list(state.get("node_log", []))

        if not sub_queries:
            return await self.retriever_agent(state)

        logger.info("[MultiRetrieve] %d sub-queries", len(sub_queries))

        async def _retrieve_one(sq: dict, idx: int) -> tuple:
            try:
                sr = await self._retriever_agent.ainvoke({"query": sq["query"]})
                docs = sr.get("documents", [])
                for d in docs:
                    d["sub_query_idx"] = idx
                    d["sub_query_text"] = sq["query"]
                return docs, sr.get("agent_log", [])
            except Exception as e:
                logger.error("[MultiRetrieve] Sub %d failed: %s", idx, e)
                return [], [f"⚠️ 子查询{idx + 1}检索失败: {e}"]

        tasks = [_retrieve_one(sq, i) for i, sq in enumerate(sub_queries)]
        results = await asyncio.gather(*tasks)

        seen_ids: set = set()
        merged: list = []
        for docs, rl in results:
            node_log.extend(rl)
            for d in docs:
                if d["id"] not in seen_ids:
                    seen_ids.add(d["id"])
                    merged.append(d)
        merged.sort(
            key=lambda d: d.get("rerank_score", d.get("rrf_score", 0)),
            reverse=True,
        )
        total = sum(len(d) for d, _ in results)
        node_log.append(
            f"🔀 并行检索: {len(sub_queries)} 子查询 → {total} 篇 → "
            f"去重后 {len(merged)} 篇"
        )
        return {"documents": merged, "node_log": node_log}

    # ====================================================================
    # ResponderAgent
    # ====================================================================

    async def responder_agent(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        documents = state.get("documents", [])
        node_log = list(state.get("node_log", []))
        sub_result = await self._responder_agent.ainvoke({
            "question": question, "documents": documents,
        })
        node_log.extend(sub_result.get("agent_log", []))
        return {
            "generation": sub_result.get("answer", ""),
            "final_answer": sub_result.get("answer", ""),
            "sources": sub_result.get("sources", []),
            "retrieval_details": sub_result.get("retrieval_details", {}),
            "node_log": node_log,
        }

    # ====================================================================
    # CriticAgent
    # ====================================================================

    async def critic(self, state: GraphState) -> Dict[str, Any]:
        sub_result = await self._critic_agent.ainvoke({
            "answer": state.get("generation", ""),
            "documents": state.get("documents", []),
            "question": state["question"],
        })
        node_log = list(state.get("node_log", []))
        node_log.extend(sub_result.get("agent_log", []))
        return {
            "critic_verdict": sub_result.get("verdict", "pass"),
            "critic_issues": sub_result.get("issues", []),
            "critic_feedback": sub_result.get("feedback", ""),
            "node_log": node_log,
        }

    # ====================================================================
    # Regenerate
    # ====================================================================

    async def regenerate(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        regenerate_count = state.get("regenerate_count", 0) + 1
        node_log = list(state.get("node_log", []))

        issues_text = "\n".join(
            f"• {i}" for i in state.get("critic_issues", [])
        )
        enhanced = (
            f"原始问题: {question}\n\n"
            f"上一版答案存在以下问题:\n{issues_text}\n\n"
            f"改进建议: {state.get('critic_feedback', '')}\n\n"
            f"请重新生成一个修正后的答案，确保:\n"
            f"1. 每个事实陈述都有源文档支撑\n"
            f"2. 所有引用 ID 准确无误\n"
            f"3. 完整回答原始问题"
        )
        node_log.append(f"🔄 修正 (第{regenerate_count}次)")

        return {
            "question": enhanced,
            "regenerate_count": regenerate_count,
            "node_log": node_log,
        }
