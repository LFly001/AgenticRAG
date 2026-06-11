"""OrchestratorAgent 子图状态 — 总调度节点的独立状态空间。"""

from typing import TypedDict, List, Dict


class OrchestratorState(TypedDict, total=False):
    """OrchestratorAgent 子图状态。

    字段说明：
    - question: 用户当前问题
    - original_question: 用户原始输入
    - session_id: 会话标识
    - chat_history: 对话历史
    - trace_id: 全链路追踪 ID（本节点初始化）
    - route_action: 路由跳转标记（本节点设置为 "intent_agent"）
    - agent_log: 执行日志
    """

    question: str
    original_question: str
    session_id: str
    chat_history: List[Dict[str, str]]
    trace_id: str
    route_action: str
    agent_log: List[str]
