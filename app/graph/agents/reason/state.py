"""ReasonAgent 子图状态 — 逻辑推理节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class ReasonState(TypedDict, total=False):
    """ReasonAgent 子图状态。

    字段说明：
    - question: 用户当前问题
    - compressed_context: 压缩后的结构化上下文（来自 ContextCompressAgent）
    - conflict_note: 文档冲突说明（来自 DocFilterAgent，为空则无冲突）
    - reasoning_draft: CoT 推理草稿，含答案逻辑框架
    - need_reretrieve: 证据是否不足，需要二次检索
    - re_retrieve_queries: 补充检索的子查询列表
    - route_action: 下一跳路由标记
    - agent_log: 执行日志
    """

    question: str
    compressed_context: str
    conflict_note: str
    reasoning_draft: str
    need_reretrieve: bool
    re_retrieve_queries: List[Dict[str, Any]]
    route_action: str
    agent_log: List[str]
