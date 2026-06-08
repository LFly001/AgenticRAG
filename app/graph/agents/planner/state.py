"""QueryPlanner 子图状态 — 查询规划专家的独立状态空间。"""
from typing import TypedDict, List, Dict


class PlannerState(TypedDict, total=False):
    """QueryPlanner 子图状态。

    字段说明：
    - question: 用户原始问题
    - is_complex: 是否需要拆解为子问题
    - complexity_reason: 复杂度判断原因
    - sub_queries: 拆解后的子问题列表，每项包含 query、strategy、priority
    - agent_log: Agent 内部执行日志
    """

    question: str
    is_complex: bool
    complexity_reason: str
    sub_queries: List[Dict[str, str]]
    agent_log: List[str]
