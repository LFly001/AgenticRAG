"""8-Agent 协作父图构建器。

图拓扑：

    START
      │
      ▼
   orchestrator（总调度，唯一入口）
      │
      │  route_action="intent_agent"（固定第一步）
      ▼
   intent_agent（意图解析 + 澄清判断）
      │
      ▼
   ┌─ route_dispatcher（全局分支分发）◄──────────────┐
   │                                                  │
   ├── "retriever_agent" → retriever_agent ────────────┤
   ├── "doc_filter_agent" → doc_filter_agent ──────────┤
   ├── "context_compress_agent" → context_compress ────┤
   ├── "reason_agent" → reason_agent ─────────────────┤
   ├── "writer_agent" → writer_agent ─────────────────┤
   ├── "anti_hallucination_agent" → anti_hallucination ┤
   └── "end" → END                                    │
                                                      │
   路由规则（route_dispatcher）：                      │
   - need_clarify=True          → END                 │
   - final_answer 已生成         → END                 │
   - need_reretrieve=True       → retriever_agent     │
   - hallucination=fail + 有余量 → reason_agent        │
   - 其他                        → 按 route_action 跳转 │
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.agents import (
    build_orchestrator_agent,
    build_intent_agent,
    build_retriever_agent,
    build_doc_filter_agent,
    build_context_compress_agent,
    build_reason_agent,
    build_writer_agent,
    build_anti_hallucination_agent,
    route_dispatcher,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 图构建入口
# ============================================================================

def build_graph(retriever=None):
    """构建 8-Agent 协作父图。

    Args:
        retriever: HybridRetriever 实例（传入 RetrieveAgent 子图，当前占位暂不使用）

    Returns:
        编译后的 CompiledGraph
    """
    # ── 构建 8 个子图 ──
    logger.info("=" * 50)
    logger.info("Building 8-Agent collaboration graph...")
    logger.info("=" * 50)

    logger.info("[1/8] OrchestratorAgent")
    orchestrator_graph = build_orchestrator_agent()

    logger.info("[2/8] IntentAgent")
    intent_graph = build_intent_agent()

    logger.info("[3/8] RetrieveAgent")
    retriever_graph = build_retriever_agent(retriever)

    logger.info("[4/8] DocFilterAgent")
    doc_filter_graph = build_doc_filter_agent()

    logger.info("[5/8] ContextCompressAgent")
    context_compress_graph = build_context_compress_agent()

    logger.info("[6/8] ReasonAgent")
    reason_graph = build_reason_agent()

    logger.info("[7/8] WriterAgent")
    writer_graph = build_writer_agent()

    logger.info("[8/8] AntiHallucinationAgent")
    anti_hallucination_graph = build_anti_hallucination_agent()

    # ── 组装父图 ──
    workflow = StateGraph(GraphState)

    # 添加 8 个节点
    workflow.add_node("orchestrator", _make_node(orchestrator_graph, "orchestrator"))
    workflow.add_node("intent_agent", _make_node(intent_graph, "intent_agent"))
    workflow.add_node("retriever_agent", _make_node(retriever_graph, "retriever_agent"))
    workflow.add_node("doc_filter_agent", _make_node(doc_filter_graph, "doc_filter_agent"))
    workflow.add_node("context_compress_agent", _make_node(context_compress_graph, "context_compress_agent"))
    workflow.add_node("reason_agent", _make_node(reason_graph, "reason_agent"))
    workflow.add_node("writer_agent", _make_node(writer_graph, "writer_agent"))
    workflow.add_node("anti_hallucination_agent", _make_node(anti_hallucination_graph, "anti_hallucination_agent"))

    # 入口：orchestrator
    workflow.set_entry_point("orchestrator")

    # 固定第一步：orchestrator → intent_agent
    workflow.add_edge("orchestrator", "intent_agent")

    # 所有后续节点 → route_dispatcher（全局分支分发）
    route_map = {
        "retriever_agent": "retriever_agent",
        "doc_filter_agent": "doc_filter_agent",
        "context_compress_agent": "context_compress_agent",
        "reason_agent": "reason_agent",
        "writer_agent": "writer_agent",
        "anti_hallucination_agent": "anti_hallucination_agent",
        "end": END,
    }

    workflow.add_conditional_edges("intent_agent", route_dispatcher, route_map)
    workflow.add_conditional_edges("retriever_agent", route_dispatcher, route_map)
    workflow.add_conditional_edges("doc_filter_agent", route_dispatcher, route_map)
    workflow.add_conditional_edges("context_compress_agent", route_dispatcher, route_map)
    workflow.add_conditional_edges("reason_agent", route_dispatcher, route_map)
    workflow.add_conditional_edges("writer_agent", route_dispatcher, route_map)
    workflow.add_conditional_edges("anti_hallucination_agent", route_dispatcher, route_map)

    compiled = workflow.compile()

    logger.info("=" * 50)
    logger.info(
        "8-Agent graph compiled (recursion_limit=50). "
        "Topology: orchestrator → intent → [route_dispatcher ↔ 6 agents] → END"
    )
    logger.info("Entry: orchestrator")
    logger.info("=" * 50)

    return compiled


# ============================================================================
# 节点包装器 — 将子图 ainvoke 暴露为父图节点 callable
# ============================================================================

def _make_node(subgraph, node_name: str):
    """将子图包装为父图可用的 async callable。

    负责：
    1. 适配父图 state → 子图 state 的字段映射
    2. 调用子图 ainvoke
    3. 将子图输出合并回父图 state（仅更新非空字段 + 合并 agent_log）
    """

    async def _node(state: GraphState):
        logger.debug("[Graph] Entering node: %s", node_name)

        # 构建子图输入（从父图 state 中选取相关字段）
        sub_input = _build_subgraph_input(state, node_name)

        # 诊断：关键数据节点记录输入大小
        if node_name == "retriever_agent":
            logger.info("[Graph.Pipe] retriever IN  raw_docs=%d", len(state.get("raw_docs", [])))
        elif node_name == "doc_filter_agent":
            logger.info("[Graph.Pipe] doc_filter IN  raw_docs=%d", len(state.get("raw_docs", [])))
        elif node_name == "context_compress_agent":
            logger.info("[Graph.Pipe] ctx_compress IN valid_docs=%d", len(state.get("valid_docs", [])))
        elif node_name == "reason_agent":
            logger.info("[Graph.Pipe] reason IN compressed_context=%d chars", len(state.get("compressed_context", "") or ""))

        # 调用子图
        sub_result = await subgraph.ainvoke(sub_input)

        # 诊断：输出大小
        if node_name == "retriever_agent":
            logger.info("[Graph.Pipe] retriever OUT raw_docs=%d", len(sub_result.get("raw_docs", [])))
        elif node_name == "doc_filter_agent":
            logger.info("[Graph.Pipe] doc_filter OUT valid_docs=%d", len(sub_result.get("valid_docs", [])))
        elif node_name == "context_compress_agent":
            ctx = sub_result.get("compressed_context", "") or ""
            logger.info("[Graph.Pipe] ctx_compress OUT compressed_context=%d chars", len(ctx))
        elif node_name == "reason_agent":
            logger.info("[Graph.Pipe] reason OUT reretrieve=%s", sub_result.get("need_reretrieve", False))

        # 合并输出
        merged = _merge_subgraph_output(state, sub_result, node_name)

        logger.debug("[Graph] Exiting node: %s, route_action=%s",
                      node_name, merged.get("route_action", "?"))
        return merged

    return _node


def _build_subgraph_input(state: GraphState, node_name: str) -> dict:
    """从父图 state 构建子图输入。

    每个 Agent 子图只需要其关心的字段，避免字段名冲突。
    """
    # 通用字段（所有子图都可能需要）
    base = {
        "question": state.get("question", ""),
        "agent_log": list(state.get("node_log", [])),
    }

    # 各节点特定字段
    extras: dict = {}

    if node_name == "orchestrator":
        extras.update({
            "session_id": state.get("session_id", ""),
            "chat_history": state.get("chat_history", []),
        })

    elif node_name == "intent_agent":
        extras.update({
            "user_query": state.get("question", ""),
            "chat_history": state.get("chat_history", []),
        })

    elif node_name == "retriever_agent":
        extras.update({
            "query_list": state.get("query_list", []),
            "re_retrieve_queries": state.get("re_retrieve_queries", []),
        })

    elif node_name == "doc_filter_agent":
        extras.update({
            "documents": state.get("raw_docs", []),
        })

    elif node_name == "context_compress_agent":
        extras.update({
            "valid_docs": state.get("valid_docs", []),
        })

    elif node_name == "reason_agent":
        extras.update({
            "compressed_context": state.get("compressed_context", ""),
            "conflict_note": state.get("conflict_note", ""),
        })

    elif node_name == "writer_agent":
        extras.update({
            "compressed_context": state.get("compressed_context", ""),
            "reasoning_draft": state.get("reasoning_draft", ""),
        })

    elif node_name == "anti_hallucination_agent":
        extras.update({
            "raw_answer": state.get("raw_answer", ""),
            "valid_docs": state.get("valid_docs", []),
        })

    return {**base, **extras}


def _merge_subgraph_output(state: GraphState, sub_result: dict, node_name: str) -> dict:
    """将子图输出合并到父图 state。

    规则：
    - 子图的 agent_log 合并到父图的 node_log
    - 空列表 / 空字符串不覆盖已有数据
    - None 值不覆盖
    - intent_agent 特殊处理：need_clarify=True 时，clarify_msg → final_answer
    """
    merged: dict = {}

    # agent_log → node_log 转换
    sub_log = sub_result.get("agent_log", [])
    if sub_log:
        old_log = list(state.get("node_log", []))
        merged["node_log"] = old_log + sub_log

    # 所有非 agent_log 字段，按非空规则合并
    for key, value in sub_result.items():
        if key == "agent_log":
            continue
        if value is None:
            continue
        if isinstance(value, (list, str, dict)) and not value:
            if key not in state or not state[key]:
                merged[key] = value
        else:
            merged[key] = value

    # intent_agent 特殊处理：闲聊 → 固定回复直接终止
    if node_name == "intent_agent" and merged.get("is_chat"):
        chat_reply = merged.get("chat_reply", "")
        if chat_reply:
            merged["final_answer"] = chat_reply
            merged["route_action"] = "end"
            logger.info(
                "[Graph] intent_agent: is_chat=True → final_answer=chat_reply"
            )

    # intent_agent 特殊处理：需要澄清 → 将澄清话术作为最终答案
    if node_name == "intent_agent" and merged.get("need_clarify"):
        clarify_msg = merged.get("clarify_msg", "")
        if clarify_msg:
            merged["final_answer"] = clarify_msg
            merged["route_action"] = "end"
            logger.info(
                "[Graph] intent_agent: need_clarify=True → final_answer=clarify_msg"
            )

    # retriever_agent 特殊处理：检索完成后重置 + 计数，避免死循环
    if node_name == "retriever_agent":
        if state.get("need_reretrieve"):
            merged["re_retrieve_count"] = state.get("re_retrieve_count", 0) + 1
        merged["need_reretrieve"] = False
        merged["re_retrieve_queries"] = []

    # anti_hallucination_agent 特殊处理（终端节点）：始终以修正后的 final_answer 作为最终输出
    if node_name == "anti_hallucination_agent":
        corrected = merged.get("final_answer", "") or state.get("raw_answer", "")
        if corrected:
            merged["final_answer"] = corrected
        merged["route_action"] = "end"
        logger.info(
            "[Graph] anti_hallucination: terminal → route_action=end, risk=%s",
            merged.get("hallucination_risk", "?"),
        )

    return merged
