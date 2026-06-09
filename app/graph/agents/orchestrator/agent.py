"""OrchestratorAgent — 总调度节点，主控唯一入口。

职责：
1. 接收用户提问与对话历史，初始化 trace_id
2. 固定流转第一步：路由至 IntentAgent
3. 全局分支分发器：监听所有节点返回标记，自动跳转对应节点

路由规则（route_dispatcher）：
    - need_clarify=True    → 直接终止，返回澄清话术
    - final_answer 已生成   → 终止（流程结束）
    - need_reretrieve=True → 跳转检索节点二次召回
    - 否则                  → 按 route_action 字段跳转

节点内部工作流（极简）：

    START
      │
      ▼
   init ──→ 初始化 trace_id、route_action="intent_agent"
      │
      ▼
     END
"""

from __future__ import annotations

import uuid
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from app.graph.agents.orchestrator.state import OrchestratorState
from app.graph.state import GraphState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 全局分支分发器（路由函数）
# ============================================================================

def route_dispatcher(state: GraphState) -> str:
    """全局分支分发器 — 根据 state 中的标记决定下一跳转节点。

    所有 Agent 节点执行完毕后都经过此函数决定下一步。

    优先级规则：
    1. need_clarify    → 终止（用户问题不明确，需澄清）
    2. final_answer    → 终止（流程结束）
    3. need_reretrieve → 重新检索
    4. 按 route_action 字段显式跳转
    """
    # 规则 1：需要澄清 → 直接终止
    if state.get("need_clarify"):
        logger.info("[Orchestrator.Route] need_clarify=True → END")
        return "end"

    # 规则 2：已有最终答案 → 终止
    if state.get("final_answer"):
        logger.info("[Orchestrator.Route] final_answer exists → END")
        return "end"

    # 规则 3：需要二次检索（最多 1 次）
    if state.get("need_reretrieve") and state.get("re_retrieve_count", 0) < 1:
        logger.info(
            "[Orchestrator.Route] need_reretrieve=True (count=%d) → retriever_agent",
            state.get("re_retrieve_count", 0),
        )
        return "retriever_agent"

    # 规则 4：按 route_action 显式跳转，但禁止二次回路无限循环
    route_action = state.get("route_action", "end")
    if route_action == "retriever_agent" and state.get("re_retrieve_count", 0) >= 1:
        logger.info(
            "[Orchestrator.Route] retriever_agent blocked (already re-retrieved) → writer_agent"
        )
        return "writer_agent"
    logger.info(
        "[Orchestrator.Route] route_action=%s", route_action
    )
    return route_action


# ============================================================================
# OrchestratorAgent 类
# ============================================================================

class OrchestratorAgent:
    """OrchestratorAgent — 总调度节点。

    作为整个 8-Agent 系统的唯一入口，负责：
    - 初始化 trace_id（全链路追踪）
    - 将 original_question 固化为用户原始输入
    - 设置首个路由标记 route_action = "intent_agent"
    """

    # ---- 节点: 初始化 ----

    async def init(self, state: OrchestratorState) -> Dict[str, Any]:
        """初始化会话与追踪信息，设置首个路由目标。

        输入（来自用户请求）：
        - question: 用户当前问题
        - session_id: 会话 ID
        - chat_history: 对话历史（若有）

        输出：
        - trace_id: 新生成的 UUID
        - original_question: 与 question 相同（固化原始输入）
        - route_action: "intent_agent"（固定第一步）
        """
        question = state.get("question", "")
        session_id = state.get("session_id", "")
        node_log: list = list(state.get("node_log", []))

        trace_id = str(uuid.uuid4())
        original_question = question  # 固化原始输入

        node_log.append(f"🚀 Orchestrator 启动 | trace_id={trace_id[:8]}")

        logger.info(
            "[Orchestrator.Init] trace=%s session=%s question=%s",
            trace_id[:8],
            session_id[:8] if session_id else "none",
            question[:80],
        )

        return {
            "trace_id": trace_id,
            "original_question": original_question,
            "route_action": "intent_agent",
            "is_chat": False,
            "chat_reply": "",
            "need_clarify": False,
            "clarify_msg": "",
            "query_list": [],
            "raw_docs": [],
            "re_retrieve_queries": [],
            "re_retrieve_count": 0,
            "need_reretrieve": False,
            "valid_docs": [],
            "conflict_note": "",
            "sources": [],
            "node_log": node_log,
        }


# ============================================================================
# 子图构建函数
# ============================================================================

def build_orchestrator_agent():
    """构建并编译 OrchestratorAgent 子图。

    Orchestrator 内部极简：init → END。
    真正的分发逻辑在父图的 route_dispatcher 条件边中完成。
    """
    agent = OrchestratorAgent()

    workflow = StateGraph(OrchestratorState)

    workflow.add_node("init", agent.init)

    workflow.set_entry_point("init")
    workflow.add_edge("init", END)

    compiled = workflow.compile()

    logger.info(
        "OrchestratorAgent subgraph compiled. "
        "Topology: START → init → END"
    )

    return compiled
