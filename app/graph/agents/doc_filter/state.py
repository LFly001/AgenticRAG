"""DocFilterAgent 子图状态 — 文档校验清洗节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class DocFilterState(TypedDict, total=False):
    """DocFilterAgent 子图状态。

    字段说明：
    - documents: 待校验的原始文档列表（来自 raw_docs）
    - _filtered: 规则过滤后的中间结果（节点间内部传递）
    - valid_docs: 校验清洗后的可信文档集合
    - conflict_note: 文档内容冲突说明（无冲突为空字符串）
    - route_action: 下一跳路由标记
    - agent_log: 执行日志
    """

    documents: List[Dict[str, Any]]
    _filtered: List[Dict[str, Any]]
    valid_docs: List[Dict[str, Any]]
    conflict_note: str
    route_action: str
    agent_log: List[str]
