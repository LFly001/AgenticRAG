"""CriticAgent 子图状态 — 答案评审专家的独立状态空间。"""
from typing import TypedDict, List


class CriticState(TypedDict, total=False):
    """CriticAgent 子图状态。

    字段说明：
    - answer: 待审查的生成答案（来自 generate 节点）
    - documents: 用于事实核查的源文档列表
    - question: 用户原始问题（用于完整性检查）
    - verdict: 评审结论（"pass" | "fail"）
    - overall_score: 1-5 的总体质量评分
    - issues: 发现的具体问题列表
    - feedback: 供 regenerate 使用的改进建议
    - agent_log: Agent 内部执行路径日志
    """

    answer: str
    documents: list
    question: str
    verdict: str
    overall_score: float
    issues: List[str]
    feedback: str
    agent_log: List[str]
