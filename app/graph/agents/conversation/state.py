"""ConversationAgent 子图状态。"""
from typing import TypedDict, List, Dict, Any


class ConversationState(TypedDict, total=False):
    """ConversationAgent 子图状态。

    字段说明：
    - question: 用户当前问题（原始输入）
    - session_id: 会话标识（空字符串 = 无记忆模式）
    - intent: 意图分类（"direct_answer" | "knowledge_query"）
    - chat_history: 从 Redis 加载的历史消息
    - enriched_question: 融合历史上下文的改写问题
    - is_followup: 是否识别为追问
    - answer: 直接回答的答案（intent=direct_answer 时填充）
    - sources: 引用来源（direct_answer 为空）
    - retrieval_details: 检索元数据
    - agent_log: Agent 内部日志
    """

    question: str
    session_id: str
    intent: str
    chat_history: List[Dict[str, str]]
    enriched_question: str
    is_followup: bool
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_details: Dict[str, Any]
    agent_log: List[str]
