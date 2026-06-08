"""LangGraph 图状态定义 — 5-Agent 协作的全流程状态流转。"""
from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict, total=False):
    """5-Agent 协作父图状态。

    ── 会话层 ──
    - question: 当前问题（ConversationAgent 可能改写为 enriched_question）
    - original_question: 用户原始输入（不变，用于保存记忆）
    - session_id: 会话标识（空字符串 = 无记忆模式）
    - chat_history: 对话历史 [{role, content}, ...]

    ── 规划层 ──
    - is_complex: 是否拆解
    - sub_queries: 子问题列表

    ── 检索层 ──
    - documents: 检索/合并去重后的文档列表

    ── 生成层 ──
    - generation: LLM 生成的答案原文
    - final_answer: 最终返回给用户的答案
    - sources: 引用来源列表
    - retrieval_details: 检索元数据

    ── 评审层 ──
    - critic_verdict / critic_issues / critic_feedback

    ── 控制层 ──
    - action: router 输出
    - regenerate_count / max_regenerates: 修正循环计数器
    - node_log: 全链路执行日志
    """

    question: str
    original_question: str
    session_id: str
    chat_history: List[Dict[str, str]]
    is_complex: bool
    sub_queries: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    generation: str
    final_answer: str
    sources: List[Dict[str, Any]]
    retrieval_details: Dict[str, Any]
    critic_verdict: str
    critic_issues: List[str]
    critic_feedback: str
    regenerate_count: int
    max_regenerates: int
    action: str
    node_log: List[str]
