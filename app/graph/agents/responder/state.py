"""ResponderAgent 子图状态 — 答案生成专家的独立状态空间。"""
from typing import TypedDict, List, Dict, Any


class ResponderState(TypedDict, total=False):
    """ResponderAgent 子图状态。

    字段说明：
    - question: 当前问题（原始或含 critic 反馈的增强版本）
    - documents: 源文档列表（来自 RetrieverAgent / multi_retrieve）
    - answer: LLM 生成的答案原文
    - sources: 引用来源列表（解析 [chunk_xxx] 得到）
    - retrieval_details: 检索元数据（doc_count, rerank_scores）
    - agent_log: Agent 内部执行日志

    内部字段（前缀 _，供节点间传递，不对外暴露）：
    - _context_str: build_context 构建的 XML 上下文
    - _source_map: parent_id → 来源详情的映射
    """

    question: str
    documents: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_details: Dict[str, Any]
    agent_log: List[str]
    # 内部字段 — build_context → generate 之间传递
    _context_str: str
    _source_map: Dict[str, Any]
