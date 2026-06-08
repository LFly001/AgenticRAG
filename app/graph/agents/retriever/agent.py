"""RetrieverAgent — 具备自主策略选择、自评估、自修正能力的检索专家子图。

内部工作流：

    START
      │
      ▼
   strategy_selector ─→ 选择: vector | bm25 | hybrid
      │
      ▼
   retrieve ─→ 按选中策略执行检索
      │
      ▼
   self_evaluate ─→ 对每篇文档评 1-5 分
      │
      ├── 平均分 ≥ 3.0 或 无剩余尝试次数
      │     │
      │     └── format_output ──→ END（返回最佳文档）
      │
      └── 平均分 < 3.0 且 还有尝试次数
            │
            └── self_rewrite ──→ 改写查询，强制 hybrid 策略
                  │
                  └──→ retrieve（循环）

与父图的关系：
    父图将 RetrieverAgent 作为黑盒节点使用，
    输入 query，输出经过自修正优化的文档列表。
"""

from __future__ import annotations

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.core.retriever import HybridRetriever
from app.graph.agents.retriever.state import RetrieverState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# LLM 工厂
# ============================================================================


def _create_strategy_llm() -> ChatOpenAI:
    """策略选择 LLM — temperature=0，路由决策必须稳定可复现。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=256,
        request_timeout=15,
        max_retries=1,
    )


def _create_evaluator_llm() -> ChatOpenAI:
    """评估 LLM — temperature=0，确保评分一致性。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=512,
        request_timeout=20,
        max_retries=1,
    )


def _create_rewriter_llm() -> ChatOpenAI:
    """改写 LLM — temperature=0.3，需要一定的语言创造性。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=512,
        request_timeout=20,
        max_retries=1,
    )


# ============================================================================
# Prompt 模板
# ============================================================================

STRATEGY_PROMPT = ChatPromptTemplate.from_template("""你是一个检索策略专家。分析用户查询的类型，选择最佳检索策略。

策略说明：
- **vector**: 适合语义模糊、概念性问题。如："公司未来的发展方向是什么？"、"如何提高客户满意度？"
- **bm25**: 适合事实精确匹配、关键词查询。如："2024年Q3财报收入"、"张三的工号是多少？"、"请假流程"
- **hybrid**: 适合混合型问题，或不确定时使用。结合向量和关键词双重优势。

请**只返回一个 JSON 对象**，格式如下：
{{"strategy": "<vector 或 bm25 或 hybrid>", "reason": "<一句话说明选择原因>"}}

Query: {query}

JSON:""")


EVALUATE_PROMPT = ChatPromptTemplate.from_template("""你是一个检索质量评估专家。评估以下文档与用户查询的相关性。

用户查询: {query}
使用策略: {strategy}

检索到的文档:
{documents}

为每篇文档打分（1-5分）：
- 5: 高度相关，直接回答查询
- 4: 相关，提供有用的上下文
- 3: 部分相关
- 2: 关联较弱
- 1: 不相关

请**只返回一个 JSON 对象**，格式如下：
{{
  "overall_score": <1-5的整数>,
  "per_doc": [
    {{
      "doc_index": 0,
      "score": 分数,
      "reason": "一句评估"
    }}
  ],
  "verdict": "<good 或 needs_improvement>",
  "diagnosis": "<irrelevant_results / too_few / outdated / other>"
}}

JSON:""")


REWRITE_PROMPT = ChatPromptTemplate.from_template("""你是一个查询改写专家。上一次检索质量不佳。

诊断信息: {diagnosis}
当前查询: {query}

请改写查询以提高检索效果：
1. 如果结果不相关，换用不同的关键词或从不同角度表述
2. 如果结果太少，使用更宽泛的词汇
3. 可以补充相关的同义词、术语、缩写/全称转换

请**只返回改写后的查询文本**，不包含任何解释。

改写后的查询:""")

# ============================================================================
# RetrieverAgent 类
# ============================================================================


class RetrieverAgent:
    """RetrieverAgent — 自主检索专家。

    封装了策略选择 → 检索 → 自评 → 改写的完整循环，
    对外表现为一个可独立调用的子图。
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever
        self._strategy_llm = _create_strategy_llm()
        self._evaluator_llm = _create_evaluator_llm()
        self._rewriter_llm = _create_rewriter_llm()

    # ---- 节点 1: 策略选择 ----

    async def strategy_selector(self, state: RetrieverState) -> Dict[str, Any]:
        """分析查询特征，选择最佳检索策略。"""
        query = state["query"]
        logger.info(f"[RetrieverAgent.Strategy] Analyzing: {query[:80]}...")

        try:
            chain = STRATEGY_PROMPT | self._strategy_llm
            response = await chain.ainvoke({"query": query})
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            strategy = result.get("strategy", "hybrid")
            reason = result.get("reason", "N/A")
        except Exception as e:
            logger.warning(f"[RetrieverAgent.Strategy] Parse failed ({e}), defaulting to hybrid.")
            strategy = "hybrid"
            reason = f"Parser error, default: {e}"

        log_entry = f"🎯 策略选择: {strategy}（{reason}）"
        logger.info(f"[RetrieverAgent.Strategy] {log_entry}")

        return {
            "strategy": strategy,
            "original_query": query,
            "attempt_count": 0,
            "max_attempts": 3,
            "agent_log": [log_entry],
        }

    # ---- 节点 2: 检索执行 ----

    async def retrieve(self, state: RetrieverState) -> Dict[str, Any]:
        """按当前策略执行检索。"""
        query = state["query"]
        strategy = state.get("strategy", "hybrid")
        attempt_count = state.get("attempt_count", 0) + 1
        agent_log = list(state.get("agent_log", []))

        logger.info(
            "[RetrieverAgent.Retrieve] Attempt %d with '%s': %s...",
            attempt_count, strategy, query[:80]
        )

        docs = await self._retriever.retrieve(query, strategy=strategy)

        log_entry = f"🔍 检索 (第{attempt_count}次, {strategy}): 获取 {len(docs)} 篇文档"
        agent_log.append(log_entry)
        logger.info(f"[RetrieverAgent.Retrieve] {log_entry}")

        return {
            "documents": docs,
            "attempt_count": attempt_count,
            "agent_log": agent_log,
        }

    # ---- 节点 3: 自我评估 ----

    async def self_evaluate(self, state: RetrieverState) -> Dict[str, Any]:
        """评估检索文档质量，判断是否需要改写重试。"""
        query = state["original_query"] or state["query"]
        strategy = state.get("strategy", "hybrid")
        documents = state.get("documents", [])
        agent_log = list(state.get("agent_log", []))

        # 无文档直接判定失败
        if not documents:
            agent_log.append("📊 自评: 无文档返回，判定为 needs_improvement")
            return {
                "agent_log": agent_log,
                "_verdict": "needs_improvement",
                "_overall_score": 0,
                "_diagnosis": "too_few",
            }

        # 构建评估上下文
        doc_summaries = []
        for i, doc in enumerate(documents):
            meta = doc.get("metadata", {})
            r_score = doc.get("rerank_score")
            r_str = f"{r_score:.4f}" if isinstance(r_score, (int, float)) else "N/A"
            doc_summaries.append(
                f"  [{i}] ID={doc['id']} | src={meta.get('source_file', '?')} | "
                f"rerank={r_str} | "
                f"text={doc.get('text', '')[:150]}..."
            )

        try:
            chain = EVALUATE_PROMPT | self._evaluator_llm
            response = await chain.ainvoke({
                "query": query,
                "strategy": strategy,
                "documents": "\n".join(doc_summaries),
            })
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            overall_score = int(result.get("overall_score", 3))
            verdict = result.get("verdict", "good")
            diagnosis = result.get("diagnosis", "")

            # 将单文档评分写入 metadata
            per_doc = result.get("per_doc", [])
            for entry in per_doc:
                idx = entry.get("doc_index", -1)
                if 0 <= idx < len(documents):
                    documents[idx]["self_eval_score"] = entry.get("score", 3)
                    documents[idx]["self_eval_reason"] = entry.get("reason", "")

        except Exception as e:
            logger.warning(f"[RetrieverAgent.Evaluate] Parse failed ({e}), defaulting to good.")
            overall_score = 3
            verdict = "good"
            diagnosis = "Evaluation error, proceeding"

        # 写入文档元数据
        for doc in documents:
            if "self_eval_score" not in doc:
                doc["self_eval_score"] = overall_score
            if "self_eval_reason" not in doc:
                doc["self_eval_reason"] = ""

        log_entry = f"📊 自评: 整体={overall_score}/5, 判定={verdict}, 诊断={diagnosis}"
        agent_log.append(log_entry)
        logger.info(f"[RetrieverAgent.Evaluate] {log_entry}")

        return {
            "documents": documents,
            "agent_log": agent_log,
            "_verdict": verdict,
            "_overall_score": overall_score,
            "_diagnosis": diagnosis,
        }

    # ---- 节点 4: 自我改写 ----

    async def self_rewrite(self, state: RetrieverState) -> Dict[str, Any]:
        """改写查询以改进检索效果。"""
        query = state["original_query"] or state["query"]
        diagnosis = state.get("_diagnosis", "unknown")
        agent_log = list(state.get("agent_log", []))

        logger.info(
            "[RetrieverAgent.Rewrite] Rewriting for: %s... (diagnosis: %s)",
            query[:80], diagnosis
        )

        try:
            chain = REWRITE_PROMPT | self._rewriter_llm
            response = await chain.ainvoke({
                "diagnosis": diagnosis,
                "query": query,
            })
            raw = response.content if hasattr(response, "content") else str(response)
            new_query = raw.strip()
        except Exception as e:
            logger.warning(f"[RetrieverAgent.Rewrite] Failed ({e}), keeping original query.")
            new_query = query

        log_entry = f"✏️ 改写查询: {new_query[:120]}"
        agent_log.append(log_entry)
        logger.info(f"[RetrieverAgent.Rewrite] {log_entry}")

        return {
            "query": new_query,
            "strategy": "hybrid",  # 改写后强制使用混合策略
            "agent_log": agent_log,
        }

    # ---- 节点 5: 格式化输出 ----

    async def format_output(self, state: RetrieverState) -> Dict[str, Any]:
        """格式化最终输出 — 按 rerank_score 排序，附带 agent 日志。"""
        documents = state.get("documents", [])
        agent_log = list(state.get("agent_log", []))

        # 按 rerank 分数降序排列
        documents_sorted = sorted(
            documents,
            key=lambda d: d.get("rerank_score", d.get("rrf_score", 0)),
            reverse=True,
        )

        agent_log.append(f"✅ 检索完成: 返回 {len(documents_sorted)} 篇文档")

        logger.info(
            f"[RetrieverAgent] Complete. "
            f"Attempts={state.get('attempt_count', 0)}, "
            f"Docs={len(documents_sorted)}"
        )

        return {
            "documents": documents_sorted,
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建函数
# ============================================================================

def build_retriever_agent(retriever: HybridRetriever):
    """构建并编译 RetrieverAgent 子图。

    Returns:
        编译后的 CompiledGraph，可通过 .ainvoke({"query": "..."}) 独立调用。
    """
    agent = RetrieverAgent(retriever)

    workflow = StateGraph(RetrieverState)

    # -- 注册节点 --
    workflow.add_node("strategy_selector", agent.strategy_selector)
    workflow.add_node("retrieve", agent.retrieve)
    workflow.add_node("self_evaluate", agent.self_evaluate)
    workflow.add_node("self_rewrite", agent.self_rewrite)
    workflow.add_node("format_output", agent.format_output)

    # -- 设置入口 --
    workflow.set_entry_point("strategy_selector")

    # -- 添加边 --
    workflow.add_edge("strategy_selector", "retrieve")
    workflow.add_edge("retrieve", "self_evaluate")

    # self_evaluate → 条件分支
    workflow.add_conditional_edges(
        "self_evaluate",
        _route_after_evaluate,
        {
            "format_output": "format_output",
            "self_rewrite": "self_rewrite",
        },
    )

    # self_rewrite → retrieve（内部修正循环）
    workflow.add_edge("self_rewrite", "retrieve")

    # format_output → END
    workflow.add_edge("format_output", END)

    compiled = workflow.compile()

    logger.info(
        "RetrieverAgent subgraph compiled. "
        "Topology: START → strategy_selector → retrieve → self_evaluate "
        "→ [format_output|self_rewrite → retrieve (loop)]"
    )

    return compiled


# ============================================================================
# 内部条件边
# ============================================================================

def _route_after_evaluate(state: RetrieverState) -> str:
    """自我评估后的路由决策。

    规则：
    1. verdict == "good" → format_output
    2. 还有剩余尝试次数 → self_rewrite
    3. 否则 → format_output（即使质量差也不再重试）
    """
    verdict = state.get("_verdict", "good")
    attempt_count = state.get("attempt_count", 0)
    max_attempts = state.get("max_attempts", 3)
    remaining = max_attempts - attempt_count

    logger.info(
        f"[RetrieverAgent.Edge] verdict={verdict}, "
        f"attempts={attempt_count}/{max_attempts}"
    )

    if verdict == "good":
        logger.info("[RetrieverAgent.Edge] → format_output (good quality)")
        return "format_output"

    if remaining > 0:
        logger.info(f"[RetrieverAgent.Edge] → self_rewrite ({remaining} attempts left)")
        return "self_rewrite"

    # 耗尽所有尝试
    logger.warning("[RetrieverAgent.Edge] → format_output (no attempts left, best effort)")
    return "format_output"
