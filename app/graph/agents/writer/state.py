"""WriterAgent 子图状态 — 答案生成节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class WriterState(TypedDict, total=False):
    """WriterAgent 子图状态。

    字段说明：
    - question: 用户当前问题
    - compressed_context: 压缩后的结构化上下文（含 doc_id 标记）
    - reasoning_draft: ReasonAgent 的推理草稿（答案逻辑框架）
    - raw_answer: 生成的完整回答（含文档引用标注 [doc_xxx]）
    - _context: 内部传递的原始压缩上下文（供引用解析使用）
    - sources: 从回答中提取的引用来源列表
    - retrieval_details: 检索元数据
    - route_action: 下一跳路由标记
    - agent_log: 执行日志
    """

    question: str
    compressed_context: str
    reasoning_draft: str
    raw_answer: str
    _context: str
    sources: List[Dict[str, Any]]
    retrieval_details: Dict[str, Any]
    route_action: str
    agent_log: List[str]
