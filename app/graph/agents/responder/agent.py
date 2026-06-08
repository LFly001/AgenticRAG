"""ResponderAgent — 答案生成专家，封装上下文构建、LLM 生成、引用解析。

内部工作流：

    START
      │
      ▼
   build_context ─→ 从 documents 构建 XML context（Redis Parent-Context 去重）
      │
      ▼
   generate ─→ LLM 生成答案（含引用标注 + 严格基于上下文约束）
      │
      ▼
   parse_citations ─→ 正则提取 [chunk_xxx]，匹配 source_map，构建 sources
      │
      ▼
   format_response ──→ END
"""

from __future__ import annotations

import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.stores.document_store import redis_store
from app.graph.agents.responder.state import ResponderState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# LLM
# ============================================================================


def _create_responder_llm() -> ChatOpenAI:
    """生成 LLM — temperature=0.1，保持事实准确性。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
        request_timeout=30,
        max_retries=2,
    )


# ============================================================================
# Prompt
# ============================================================================

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """你是一个企业级智能助手。请根据以下提供的上下文信息（Context）来回答用户的问题（Question）。

<context>
{context}
</context>

要求：
1. **严格基于上下文**：如果上下文中没有足够信息回答问题，请诚实回答"根据现有知识库，无法找到相关答案"，严禁编造事实。
2. **引用规范**：在回答中的每个关键事实陈述后，必须标注信息来源的文档 ID。
   - 格式必须为方括号包裹 ID，例如：[doc_123]
   - ID 必须与 <doc_id> 标签中的内容完全一致。
   - 如果一句话参考了多个文档，请全部列出，例如：[doc_1][doc_2]
3. **语言风格**：回答必须准确、简洁、专业，使用与用户提问相同的语言。

Question: {question}

Answer:"""
)

# ============================================================================
# ResponderAgent
# ============================================================================


class ResponderAgent:
    """ResponderAgent — 答案生成专家。

    封装 上下文构建 → LLM 生成 → 引用解析 → 格式化输出 四个节点，
    对外表现为一个可独立调用的子图。
    """

    def __init__(self) -> None:
        self._llm = _create_responder_llm()
        self._prompt = GENERATE_PROMPT

    # ---- 节点 1: 构建上下文 ----

    async def build_context(self, state: ResponderState) -> Dict[str, Any]:
        """从 documents 构建 XML 格式上下文（含 Redis Parent-Context 去重）。"""
        documents = state.get("documents", [])
        agent_log = list(state.get("agent_log", []))

        if not documents:
            agent_log.append("📝 无文档，将返回默认响应")
            return {
                "_context_str": "",
                "_source_map": {},
                "agent_log": agent_log,
            }

        context_parts: list[str] = []
        source_map: dict = {}
        seen_parents: set = set()

        for doc in documents:
            meta = doc.get("metadata", {})
            parent_id = meta.get("parent_id")

            if parent_id and parent_id not in seen_parents:
                parent_text = redis_store.get_parent_context(parent_id)
                if not parent_text:
                    logger.warning(
                        "Parent context not found in Redis for %s, "
                        "using child text.",
                        parent_id,
                    )
                    parent_text = doc.get("text", "")

                seen_parents.add(parent_id)
                src = meta.get("source_file", "Unknown")
                pg = meta.get("page_number", "N/A")

                context_parts.append(
                    f"<doc>\n"
                    f"<doc_id>{parent_id}</doc_id>\n"
                    f"<source_file>{src}</source_file>\n"
                    f"<page>{pg}</page>\n"
                    f"<content>{parent_text}</content>\n"
                    f"</doc>\n"
                )

                source_map[parent_id] = {
                    "id": parent_id,
                    "source_file": src,
                    "page": str(pg),
                    "type": meta.get("element_type", "Text"),
                    "snippet": (
                        parent_text[:150] + "..."
                        if len(parent_text) > 150
                        else parent_text
                    ),
                }

        context_str = "\n".join(context_parts)
        agent_log.append(
            f"📝 构建上下文: {len(seen_parents)} 个唯一父文档"
        )

        return {
            "_context_str": context_str,
            "_source_map": source_map,
            "agent_log": agent_log,
        }

    # ---- 节点 2: LLM 生成 ----

    async def generate(self, state: ResponderState) -> Dict[str, Any]:
        """调用 LLM 生成答案。"""
        question = state["question"]
        context_str = state.get("_context_str", "")
        agent_log = list(state.get("agent_log", []))

        if not context_str:
            raw_answer = "未检索到相关文档，无法回答。"
            agent_log.append("📝 生成: 无上下文，返回默认响应")
            return {"answer": raw_answer, "agent_log": agent_log}

        try:
            logger.info(
                "[ResponderAgent] Generating for: %s...", question[:80]
            )
            chain = self._prompt | self._llm
            response = await chain.ainvoke({
                "context": context_str,
                "question": question,
            })
            raw_answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as e:
            logger.error("LLM Generation Error: %s", e, exc_info=True)
            raw_answer = "抱歉，连接人工智能服务时发生错误，请稍后重试。"

        agent_log.append(f"📝 生成完成: {len(raw_answer)} 字符")

        return {"answer": raw_answer, "agent_log": agent_log}

    # ---- 节点 3: 引用解析 ----

    async def parse_citations(self, state: ResponderState) -> Dict[str, Any]:
        """从 LLM 输出中提取 [chunk_xxx] 引用，匹配 source_map。"""
        raw_answer = state.get("answer", "")
        source_map: dict = state.get("_source_map", {})
        agent_log = list(state.get("agent_log", []))

        cited_ids = list(
            set(re.findall(r'\[(chunk_[a-zA-Z0-9_]+)\]', raw_answer))
        )

        final_sources = [
            source_map[cid] for cid in cited_ids if cid in source_map
        ]

        if not final_sources and source_map:
            logger.warning(
                "LLM did not cite any documents. Returning fallback sources."
            )
            final_sources = list(source_map.values())[:3]

        agent_log.append(
            f"📎 引用解析: {len(cited_ids)} 个引用 → "
            f"{len(final_sources)} 个有效来源"
        )

        documents = state.get("documents", [])
        doc_count_value = len({d.get("id") for d in documents})

        return {
            "sources": final_sources,
            "retrieval_details": {
                "doc_count": doc_count_value,
                "rerank_scores": [
                    float(d.get("rerank_score", 0)) for d in documents
                ],
            },
            "agent_log": agent_log,
        }

    # ---- 节点 4: 格式化 ----

    async def format_response(self, state: ResponderState) -> Dict[str, Any]:
        """格式化最终输出。"""
        answer = state.get("answer", "")
        sources = state.get("sources", [])
        agent_log = list(state.get("agent_log", []))

        agent_log.append(
            f"✅ 响应完成: {len(answer)} 字符, "
            f"{len(sources)} 个来源"
        )

        return {"agent_log": agent_log}


# ============================================================================
# 子图构建
# ============================================================================


def build_responder_agent():
    """构建并编译 ResponderAgent 子图。"""
    agent = ResponderAgent()

    workflow = StateGraph(ResponderState)

    workflow.add_node("build_context", agent.build_context)
    workflow.add_node("generate", agent.generate)
    workflow.add_node("parse_citations", agent.parse_citations)
    workflow.add_node("format_response", agent.format_response)

    workflow.set_entry_point("build_context")
    workflow.add_edge("build_context", "generate")
    workflow.add_edge("generate", "parse_citations")
    workflow.add_edge("parse_citations", "format_response")
    workflow.add_edge("format_response", END)

    compiled = workflow.compile()

    logger.info(
        "ResponderAgent subgraph compiled. "
        "Topology: START → build_context → generate → "
        "parse_citations → format_response → END"
    )

    return compiled
