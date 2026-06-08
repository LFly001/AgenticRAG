"""多 Agent 协作模块 — 5-Agent 协作架构。

Agent 清单：
┌────────────────────┬──────────────────────────────────┐
│ Agent              │ 职责                              │
├────────────────────┼──────────────────────────────────┤
│ ConversationAgent  │ 对话记忆 + 追问识别 + 上下文融合    │
│ QueryPlanner       │ 复杂度判断 + 复合问题拆解为子查询   │
│ RetrieverAgent     │ 策略选择 → 检索 → 自评 → 改写      │
│ ResponderAgent     │ 上下文构建 → LLM 生成 → 引用解析    │
│ CriticAgent        │ 事实核查 → 引用验证 → 完整性评估    │
└────────────────────┴──────────────────────────────────┘
"""

from app.graph.agents.conversation.agent import (
    build_conversation_agent, ConversationAgent,
)
from app.graph.agents.conversation.state import ConversationState
from app.graph.agents.retriever.agent import build_retriever_agent, RetrieverAgent
from app.graph.agents.retriever.state import RetrieverState
from app.graph.agents.critic.agent import build_critic_agent, CriticAgent
from app.graph.agents.critic.state import CriticState
from app.graph.agents.planner.agent import build_planner_agent, QueryPlanner
from app.graph.agents.planner.state import PlannerState
from app.graph.agents.responder.agent import build_responder_agent, ResponderAgent
from app.graph.agents.responder.state import ResponderState

__all__ = [
    "build_conversation_agent", "ConversationAgent", "ConversationState",
    "build_retriever_agent", "RetrieverAgent", "RetrieverState",
    "build_critic_agent", "CriticAgent", "CriticState",
    "build_planner_agent", "QueryPlanner", "PlannerState",
    "build_responder_agent", "ResponderAgent", "ResponderState",
]
