"""LangGraph 父图构建器 — 5-Agent 协作顶层编排。

图拓扑（简化后）：

    START
      │
      ▼
   conversation_agent
      │
      ├── (direct_answer) ──► save_conv → END
      │
      └── (knowledge_query) → query_planner
                                │
           ┌────────────────────┴────────────────────┐
           │                                         │
      (simple)                                   (complex)
           │                                         │
           ▼                                         ▼
      retriever_agent                        multi_retrieve
           │                                  (parallel fan-out)
           └────────────────┬──────────────────────┘
                            ▼
                      responder_agent ──► critic
                                            │
                                  ┌─────────┴─────────┐
                              (pass)              (fail)
                                  │                   │
                                  ▼                   ▼
                             save_conv → END      regenerate
                                                    │
                                                    └──► responder_agent

5-Agent 矩阵：
┌──────────────────┬──────────────────────────────────────┬──────────┐
│ Agent            │ 职责                                  │ 类型     │
├──────────────────┼──────────────────────────────────────┼──────────┤
│ ConversationAgent│ 意图分类 + 对话记忆 + 追问融合 + 闲聊回复│ 子图     │
│ QueryPlanner     │ 简单 vs 复合 + 拆解子问题              │ 子图     │
│ RetrieverAgent   │ 策略选择 → 检索 → 自评 → 改写         │ 子图 ×N  │
│ ResponderAgent   │ 上下文构建 → LLM 生成 → 引用解析       │ 子图     │
│ CriticAgent      │ 事实核查 + 引用验证 + 完整性评估       │ 子图     │
└──────────────────┴──────────────────────────────────────┴──────────┘
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import GraphNodes
from app.graph.agents import (
    build_conversation_agent,
    build_planner_agent,
    build_retriever_agent,
    build_responder_agent,
    build_critic_agent,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================
# 条件边
# ===========================

def _route_after_conversation(state: GraphState) -> str:
    """闲聊已生成答案 → save_conv；知识问答 → query_planner。"""
    if state.get("final_answer"):
        return "save_conversation"
    return "query_planner"


def _route_after_planner(state: GraphState) -> str:
    if state.get("is_complex"):
        return "multi_retrieve"
    return "retriever_agent"


def _route_after_critic(state: GraphState) -> str:
    verdict = state.get("critic_verdict", "pass")
    remaining = state.get("max_regenerates", 2) - state.get("regenerate_count", 0)
    if verdict == "pass":
        return "save_conversation"
    if remaining > 0:
        return "regenerate"
    return "save_conversation"


# ===========================
# 图构建入口
# ===========================

def build_graph(retriever):
    logger.info("Building ConversationAgent subgraph...")
    conversation_graph = build_conversation_agent()

    logger.info("Building QueryPlanner subgraph...")
    planner_graph = build_planner_agent()

    logger.info("Building RetrieverAgent subgraph...")
    retriever_agent_graph = build_retriever_agent(retriever)

    logger.info("Building ResponderAgent subgraph...")
    responder_agent_graph = build_responder_agent()

    logger.info("Building CriticAgent subgraph...")
    critic_agent_graph = build_critic_agent()

    nodes = GraphNodes(
        conversation_graph,
        planner_graph,
        retriever_agent_graph,
        responder_agent_graph,
        critic_agent_graph,
    )

    workflow = StateGraph(GraphState)

    workflow.add_node("conversation_agent", nodes.conversation_agent)
    workflow.add_node("query_planner", nodes.query_planner)
    workflow.add_node("retriever_agent", nodes.retriever_agent)
    workflow.add_node("multi_retrieve", nodes.multi_retrieve)
    workflow.add_node("responder_agent", nodes.responder_agent)
    workflow.add_node("critic", nodes.critic)
    workflow.add_node("regenerate", nodes.regenerate)
    workflow.add_node("save_conversation", nodes.save_conversation)

    workflow.set_entry_point("conversation_agent")

    # 闲聊: conversation → save → END
    # 知识: conversation → query_planner → ...
    workflow.add_conditional_edges(
        "conversation_agent", _route_after_conversation,
        {"save_conversation": "save_conversation", "query_planner": "query_planner"},
    )
    workflow.add_edge("save_conversation", END)

    # 检索路径
    workflow.add_conditional_edges(
        "query_planner", _route_after_planner,
        {"retriever_agent": "retriever_agent", "multi_retrieve": "multi_retrieve"},
    )
    workflow.add_edge("retriever_agent", "responder_agent")
    workflow.add_edge("multi_retrieve", "responder_agent")
    workflow.add_edge("responder_agent", "critic")

    # 评审路径: pass → save → END, fail → regenerate → responder
    workflow.add_conditional_edges(
        "critic", _route_after_critic,
        {"save_conversation": "save_conversation", "regenerate": "regenerate"},
    )
    workflow.add_edge("regenerate", "responder_agent")

    compiled = workflow.compile()

    logger.info(
        "5-Agent graph compiled (5 subgraphs, 8 nodes). "
        "Entry: conversation_agent"
    )
    return compiled
