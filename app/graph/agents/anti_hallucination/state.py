"""AntiHallucinationAgent 子图状态 — 幻觉检测节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class AntiHallucinationState(TypedDict, total=False):
    """AntiHallucinationAgent 子图状态。

    字段说明：
    - question: 用户原始问题
    - raw_answer: WriterAgent 生成的待检测答案
    - valid_docs: DocFilterAgent 输出的可信文档集合（事实依据）
    - final_answer: 修正后的最终可信答案
    - hallucination_risk: 幻觉风险等级（"none" | "mild" | "high"）
    - _issues_found: 核查发现的问题列表（节点间传递）
    - _corrections_made: 已修正的内容列表（节点间传递）
    - route_action: 下一跳路由标记
    - agent_log: 执行日志
    """

    question: str
    raw_answer: str
    valid_docs: List[Dict[str, Any]]
    final_answer: str
    hallucination_risk: str
    _issues_found: List[str]
    _corrections_made: List[str]
    route_action: str
    agent_log: List[str]
