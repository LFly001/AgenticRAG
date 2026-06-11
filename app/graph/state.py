"""8-Agent 协作父图状态 — 全链路字段定义。

Agent 清单：
┌──────────────────────┬──────────────────────────────────┐
│ Agent                │ 职责                              │
├──────────────────────┼──────────────────────────────────┤
│ OrchestratorAgent    │ 总调度，唯一入口，全局分支分发      │
│ IntentAgent          │ 意图解析，澄清判断                 │
│ RetrieveAgent        │ 检索调度，多策略召回               │
│ DocFilterAgent       │ 文档校验清洗，去重过滤             │
│ ContextCompressAgent │ 上下文压缩，token 预算管理         │
│ ReasonAgent          │ 逻辑推理，思维链生成               │
│ WriterAgent          │ 答案生成，引用规范                 │
│ AntiHallucinationAgent│ 幻觉检测，事实校验                │
└──────────────────────┴──────────────────────────────────┘
"""

from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict, total=False):
    """8-Agent 协作父图状态。

    ── 会话层 ──
    - question: 用户当前问题（可能被上游改写）
    - original_question: 用户原始输入（不变，用于保存记忆）
    - session_id: 会话标识（空字符串 = 单轮模式）
    - chat_history: 对话历史 [{role, content}, ...]
    - trace_id: 全链路追踪 ID（Orchestrator 初始化）

    ── 意图层 ──
    - is_chat: 是否为闲聊（短路标记）
    - chat_reply: 闲聊固定回复
    - query_list: 拆解后的子查询列表 [{query}, ...]
    - need_clarify: 是否需要向用户澄清问题
    - clarify_msg: 澄清话术文本

    ── 检索层 ──
    - raw_docs: 并行检索 + RRF 融合后的原始文档列表
    - re_retrieve_queries: 二次检索的查询列表（下游触发时填充）
    - need_reretrieve: 是否需要二次检索

    ── 文档过滤层 ──
    - valid_docs: 校验清洗后的可信文档集合
    - conflict_note: 文档内容冲突说明（无冲突为空字符串）

    ── 上下文压缩层 ──
    - compressed_context: 压缩后的上下文字符串（供 ReasonAgent 使用）

    ── 推理层 ──
    - reasoning_draft: CoT 推理草稿，含答案逻辑框架

    ── 生成层 ──
    - raw_answer: 生成的完整答案（含引用标注）
    - final_answer: 最终返回给用户的答案（raw_answer 经幻觉检测通过后确认）
    - sources: 引用来源列表 [{id, source_file, page, type, snippet}, ...]
    - retrieval_details: 检索元数据 {doc_count, rerank_scores, ...}

    ── 幻觉检测层 ──
    - hallucination_risk: 幻觉风险等级（"none" | "mild" | "high"）

    ── 控制层 ──
    - route_action: 路由跳转标记，Orchestrator 据此分发到下一节点
    - node_log: 全链路执行日志
    """

    # 会话层
    question: str
    original_question: str
    session_id: str
    trace_id: str

    # 意图层
    is_chat: bool
    chat_reply: str
    query_list: List[Dict[str, Any]]
    need_clarify: bool
    clarify_msg: str

    # 检索层
    raw_docs: List[Dict[str, Any]]
    re_retrieve_queries: List[Dict[str, Any]]
    re_retrieve_count: int
    need_reretrieve: bool

    # 文档过滤层
    valid_docs: List[Dict[str, Any]]
    conflict_note: str

    # 上下文压缩层
    compressed_context: str

    # 推理层
    reasoning_draft: str

    # 生成层
    raw_answer: str
    final_answer: str
    sources: List[Dict[str, Any]]
    retrieval_details: Dict[str, Any]

    # 幻觉检测层
    hallucination_risk: str

    # 控制层
    route_action: str
    node_log: List[str]
