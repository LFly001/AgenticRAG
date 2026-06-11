"""RetrieveAgent 子图状态 — 检索调度节点的独立状态空间。"""

from typing import TypedDict, List, Dict, Any


class RetrieverState(TypedDict, total=False):
    """RetrieveAgent 子图状态。

    字段说明：
    - query_list: 正常检索的子查询列表 [{query}, ...]
    - re_retrieve_queries: 二次检索的查询列表（与 query_list 互斥，优先使用）
    - _retrieve_results: 并行检索的原始结果（节点间内部传递）
    - raw_docs: 并行检索 + RRF 融合 + 去重排序后的文档列表
    - retrieval_details: 检索元数据 {doc_count, rerank_scores}
    - route_action: 下一跳路由标记
    - agent_log: 执行日志
    """

    query_list: List[Dict[str, Any]]
    re_retrieve_queries: List[Dict[str, Any]]
    _retrieve_results: List[Any]
    raw_docs: List[Dict[str, Any]]
    retrieval_details: Dict[str, Any]
    route_action: str
    agent_log: List[str]
