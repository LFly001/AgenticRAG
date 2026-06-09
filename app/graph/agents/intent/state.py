"""IntentAgent 子图状态 — 意图解析节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class IntentState(TypedDict, total=False):
    """IntentAgent 子图状态。

    字段说明：
    - user_query: 用户当前问题（原始输入）
    - chat_history: 对话历史 [{role, content}, ...]
    - is_chat: 是否为闲聊（问候/感谢/告别/无关话题）
    - chat_reply: 闲聊固定回复
    - resolved_query: 指代消解后的完整问题（内部传递）
    - query_list: 拆解后的子查询列表 [{query}, ...]
    - need_clarify: 是否需要向用户澄清
    - clarify_msg: 澄清话术
    - route_action: 下一跳路由标记
    - agent_log: 执行日志
    """

    user_query: str
    chat_history: List[Dict[str, str]]
    is_chat: bool
    chat_reply: str
    resolved_query: str
    query_list: List[Dict[str, Any]]
    need_clarify: bool
    clarify_msg: str
    route_action: str
    agent_log: List[str]
