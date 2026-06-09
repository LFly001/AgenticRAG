"""ContextCompressAgent 子图状态 — 上下文压缩节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class ContextCompressState(TypedDict, total=False):
    """ContextCompressAgent 子图状态。

    字段说明：
    - question: 用户问题（用于相关性导向压缩）
    - valid_docs: 过滤后的可信文档列表（来自 DocFilterAgent）
    - compressed_context: 压缩后的结构化上下文字符串
    - route_action: 下一跳路由标记
    - agent_log: 执行日志

    内部字段（前缀 _）：
    - _raw_context: 格式化但未压缩的原始上下文
    - _token_estimate: 预估 token 数
    """

    question: str
    valid_docs: List[Dict[str, Any]]
    compressed_context: str
    route_action: str
    agent_log: List[str]
    # 内部传递
    _raw_context: str
    _token_estimate: int
