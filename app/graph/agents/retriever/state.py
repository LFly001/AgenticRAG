"""RetrieverAgent 子图状态 — 检索专家的独立状态空间。"""
from typing import TypedDict, List, Dict, Any


class RetrieverState(TypedDict, total=False):
    """RetrieverAgent 子图状态。

    字段说明：
    - query: 当前要检索的查询文本（可能被 self_rewrite 修改）
    - original_query: 用户原始查询（用于最终溯源和评估基准）
    - strategy: 当前选择的检索策略（vector / bm25 / hybrid）
    - documents: 当前检索到的文档列表
    - attempt_count: 已执行检索尝试的次数
    - max_attempts: 最大允许尝试次数（默认 3），防止无限循环
    - agent_log: Agent 内部执行路径日志

    内部路由信号（由 self_evaluate 写入，_route_after_evaluate 读取）：
    - _verdict: "good" | "needs_improvement"
    - _overall_score: 1-5 的整体检索质量评分
    - _diagnosis: 诊断原因（irrelevant_results / too_few / outdated / other）
    """

    query: str
    original_query: str
    strategy: str
    documents: List[Dict[str, Any]]
    attempt_count: int
    max_attempts: int
    agent_log: List[str]
