"""RetrieveAgent — 检索调度节点。

职责：
1. 遍历 query_list / re_retrieve_queries，并行执行 BM25 + 向量混合检索
2. RRF 融合多路召回结果（由 HybridRetriever 内部完成）
3. 跨子查询去重合并，按 rerank_score 降序排序
4. 统一汇总输出 raw_docs

内部工作流（2 节点）：

    START
      │
      ▼
   parallel_retrieve ─→ 遍历所有子查询，asyncio.gather 并行检索
      │
      ▼
   merge_and_format ─→ 跨子查询去重 + 排序 + 汇总 → raw_docs
      │
      ▼
     END

下游固定节点：doc_filter_agent
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from app.core.retriever import HybridRetriever
from app.graph.agents.retriever.state import RetrieverState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# RetrieveAgent 类
# ============================================================================

class RetrieveAgent:
    """RetrieveAgent — 检索调度专家。

    对每个子查询独立执行混合检索（向量 + BM25 + RRF + CrossEncoder 重排序），
    并行化以降低延迟，最后跨子查询去重汇总。

    依赖：
        HybridRetriever — 封装了向量/Bm25/RRF/重排序全部逻辑
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    # ---- 节点 1: 并行检索 ----

    async def parallel_retrieve(self, state: RetrieverState) -> Dict[str, Any]:
        """遍历所有子查询，并行执行混合检索。

        优先级：
        1. 若 re_retrieve_queries 非空 → 二次检索模式
        2. 否则 → 正常检索，使用 query_list
        """
        agent_log: list[str] = list(state.get("agent_log", []))

        # 确定查询来源
        re_queries: List[Dict[str, Any]] = state.get("re_retrieve_queries", [])
        if re_queries:
            queries = re_queries
            mode = "re_retrieve"
            agent_log.append(f"🔍 二次检索模式: {len(queries)} 个查询")
        else:
            queries = state.get("query_list", [])
            mode = "normal"
            if not queries:
                agent_log.append("⚠️ query_list 为空，跳过检索")
                return {
                    "_retrieve_results": [],
                    "agent_log": agent_log,
                }
            agent_log.append(f"🔍 检索模式: {len(queries)} 个子查询")

        logger.info(
            "[RetrieveAgent] %s mode, %d sub-queries", mode, len(queries)
        )

        # 并行检索
        async def _retrieve_one(sq: dict, idx: int) -> tuple:
            """对单个子查询执行检索，捕获异常。"""
            query_text = sq.get("query", "")

            try:
                logger.debug(
                    "[RetrieveAgent] Sub %d: %s",
                    idx, query_text[:60],
                )
                docs = await self._retriever.retrieve(query=query_text)
                # 标注来源子查询
                for d in docs:
                    d["sub_query_idx"] = idx
                    d["sub_query_text"] = query_text
                return (idx, docs, None)
            except Exception as e:
                logger.error(
                    "[RetrieveAgent] Sub %d failed: %s", idx, e
                )
                return (idx, [], str(e))

        tasks = [_retrieve_one(sq, i) for i, sq in enumerate(queries)]
        results: List[tuple] = await asyncio.gather(*tasks)

        # 汇总原始结果（不做去重，留给 merge 节点）
        all_results: list = []
        has_errors = False
        for idx, docs, err in results:
            if err:
                agent_log.append(f"   ⚠️ Q{idx+1} 检索失败: {err}")
                has_errors = True
            else:
                agent_log.append(
                    f"   Q{idx+1}: {len(docs)} 篇 → \"{queries[idx].get('query', '')[:50]}\""
                )
            all_results.append(docs)

        total_before_dedup = sum(len(d) for d in all_results)
        agent_log.append(
            f"🔍 检索完成: {len(queries)} 子查询 → 合计 {total_before_dedup} 篇（去重前）"
        )

        logger.info(
            "[RetrieveAgent] %d sub-queries → %d docs (pre-dedup), errors=%s",
            len(queries), total_before_dedup, has_errors,
        )

        return {
            "_retrieve_results": all_results,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 去重 + 排序 + 汇总 ----

    async def merge_and_format(self, state: RetrieverState) -> Dict[str, Any]:
        """跨子查询去重合并，按 rerank_score 降序输出 raw_docs。

        去重规则：同一 doc.id 在多个子查询命中时，保留最高 rerank_score 的那条。
        排序规则：优先 rerank_score，其次 rrf_score。
        """
        all_results: list = state.get("_retrieve_results", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        if not all_results:
            agent_log.append("📭 无检索结果，raw_docs 为空")
            return {
                "raw_docs": [],
                "retrieval_details": {"doc_count": 0, "rerank_scores": []},
                "route_action": "doc_filter_agent",
                "agent_log": agent_log,
            }

        # 跨子查询去重：同一 doc_id 保留最高 rerank_score
        seen: dict[str, dict] = {}
        for doc_list in all_results:
            for doc in doc_list:
                doc_id = doc.get("id", "")
                if not doc_id:
                    continue
                current_score = doc.get("rerank_score", doc.get("rrf_score", 0))
                if doc_id not in seen:
                    seen[doc_id] = doc
                else:
                    existing_score = seen[doc_id].get(
                        "rerank_score", seen[doc_id].get("rrf_score", 0)
                    )
                    if current_score > existing_score:
                        seen[doc_id] = doc

        # 按 rerank_score 降序排列
        raw_docs = sorted(
            seen.values(),
            key=lambda d: d.get("rerank_score", d.get("rrf_score", 0)),
            reverse=True,
        )

        total_before = sum(len(r) for r in all_results)
        dupes_removed = total_before - len(raw_docs)

        agent_log.append(
            f"📊 汇总: {total_before} 篇 → 去重 {dupes_removed} → "
            f"最终 {len(raw_docs)} 篇"
        )

        if raw_docs:
            # 输出 Top-3 摘要
            for i, doc in enumerate(raw_docs[:3]):
                score = doc.get("rerank_score", doc.get("rrf_score", 0))
                meta = doc.get("metadata", {})
                src = meta.get("source_file", "?")
                snippet = (doc.get("text", "")[:60]).replace("\n", " ")
                agent_log.append(
                    f"   #{i+1} [{score:.3f}] {src}: {snippet}..."
                )

        logger.info(
            "[RetrieveAgent] Merge: %d → %d docs (removed %d dupes)",
            total_before, len(raw_docs), dupes_removed,
        )

        return {
            "raw_docs": raw_docs,
            "retrieval_details": {
                "doc_count": len(raw_docs),
                "rerank_scores": [
                    d.get("rerank_score", d.get("rrf_score", 0))
                    for d in raw_docs
                ],
            },
            "route_action": "doc_filter_agent",
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建函数
# ============================================================================

def build_retriever_agent(retriever: Optional[HybridRetriever] = None):
    """构建并编译 RetrieveAgent 子图。

    Args:
        retriever: HybridRetriever 实例。若为 None，使用占位模式（返回空）。

    Returns:
        编译后的 CompiledGraph

    拓扑：
        START → parallel_retrieve → merge_and_format → END
    """
    agent = RetrieveAgent(retriever) if retriever else _PlaceholderRetriever()

    workflow = StateGraph(RetrieverState)

    workflow.add_node("parallel_retrieve", agent.parallel_retrieve)
    workflow.add_node("merge_and_format", agent.merge_and_format)

    workflow.set_entry_point("parallel_retrieve")
    workflow.add_edge("parallel_retrieve", "merge_and_format")
    workflow.add_edge("merge_and_format", END)

    compiled = workflow.compile()
    return compiled


# ============================================================================
# 占位回退（retriever 不可用时）
# ============================================================================

class _PlaceholderRetriever:
    """无检索器时的安全回退。"""

    async def parallel_retrieve(self, state: RetrieverState) -> Dict[str, Any]:
        agent_log = list(state.get("agent_log", []))
        agent_log.append("⚠️ [RetrieveAgent] 无 HybridRetriever 实例，返回空")
        return {"_retrieve_results": [], "agent_log": agent_log}

    async def merge_and_format(self, state: RetrieverState) -> Dict[str, Any]:
        agent_log = list(state.get("agent_log", []))
        agent_log.append("📭 占位模式: raw_docs 为空")
        return {
            "raw_docs": [],
            "retrieval_details": {"doc_count": 0, "rerank_scores": []},
            "route_action": "doc_filter_agent",
            "agent_log": agent_log,
        }
