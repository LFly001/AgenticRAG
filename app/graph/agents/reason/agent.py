"""ReasonAgent — 逻辑推理节点。

职责：
1. 基于 compressed_context 做 CoT 链式推理，搭建答案逻辑框架 → reasoning_draft
2. 判断证据充足度：
   - 不足 → need_reretrieve=True + 生成 re_retrieve_queries
   - 充足 → need_reretrieve=False
3. 存在文档冲突则在推理草稿中备注冲突点

内部工作流（2 节点）：

    START
      │
      ▼
   cot_reason ─→ LLM CoT 推理 + 证据充足度判断
      │
      ▼
   finalize ─→ 根据 need_reretrieve 设置 route_action
      │
      ▼
     END

路由分支（由父图 route_dispatcher 处理）：
    - need_reretrieve=True  → 跳转 retriever_agent 二次检索
    - need_reretrieve=False → 下游 writer_agent
"""

from __future__ import annotations

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.graph.agents.reason.state import ReasonState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# LLM
# ============================================================================

def _create_reason_llm() -> ChatOpenAI:
    """推理 LLM — temperature=0.1，给 CoT 留一点多样性空间。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0.1,
        max_tokens=2048,
        request_timeout=45,
        max_retries=2,
    )


# ============================================================================
# Prompt
# ============================================================================

COT_REASON_PROMPT = ChatPromptTemplate.from_template("""你是一个严谨的逻辑推理专家。请基于提供的上下文信息，对用户问题进行链式推理（Chain-of-Thought），搭建答案的逻辑框架。

<用户问题>
{question}
</用户问题>

<上下文信息>
{context}
</上下文信息>

<已知冲突>
{conflict_note}
</已知冲突>

## 任务步骤

### 步骤 1：链式推理（CoT）
按以下结构逐层推理：
1. **分解问题**：将问题拆解为核心子问题
2. **证据提取**：从上下文中提取每个子问题相关的关键事实（标注来源 doc_id）
3. **逻辑推导**：基于提取的事实，逐步推导出结论
4. **冲突处理**：如果「已知冲突」不为空，在推理中注明存在矛盾的证据，标记为 "⚠️ 冲突"

### 步骤 2：证据充足度判断
检查是否具备回答问题的完整证据链：
- **证据不足的条件**（同时满足以下才判定为不足）：
  - 关键信息缺失：上下文中完全没有与问题相关的政策、规则、数据
  - 无法推导：即使通过逻辑推导也无法从现有上下文得出答案
- **以下情况应判定为证据充足**：
  - 上下文中包含相关政策/规则条款，可以直接套用计算（如工龄区间对应天数）
  - 上下文中包含同类信息可作参考推断
  - 问题中的具体人名/时间只是参数，规则本身存在即可回答
  - 即使存在冲突，只要冲突各方都有证据，也视为充足（但需在推理中注明）

### 步骤 3：生成补充检索（仅当证据不足时）
- 针对缺失的信息维度，生成 1-3 个**与原始问法不同**的子查询
- 使用不同关键词、同义词、缩写/全称变体，尝试从不同角度命中文档
- **严禁**直接复制原始问题文本作为子查询
- 如果确实想不出有效替代查询，设 need_reretrieve=false 放弃重检索

## 输出格式

请**只返回一个 JSON 对象**：

{{
  "reasoning_draft": "完整的链式推理过程文本...",
  "need_reretrieve": false,
  "re_retrieve_queries": []
}}

如果 need_reretrieve=true，re_retrieve_queries 格式为：
[
  {{"query": "补充检索查询1"}},
  {{"query": "补充检索查询2"}}
]

reasoning_draft 应包含清晰的推理步骤，方便下游 WriterAgent 据此生成最终答案。

JSON:""")


# ============================================================================
# ReasonAgent 类
# ============================================================================

class ReasonAgent:
    """ReasonAgent — 逻辑推理专家。

    CoT 链式推理 + 证据充足度判断 + 触发二次检索。
    """

    def __init__(self) -> None:
        self._reason_llm = _create_reason_llm()

    # ---- 节点 1: CoT 推理 ----

    async def cot_reason(self, state: ReasonState) -> Dict[str, Any]:
        """LLM CoT 推理 + 证据充足度判断。"""
        question = state.get("question", "")
        compressed_context = state.get("compressed_context", "")
        conflict_note = state.get("conflict_note", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        # 空上下文 → 直接判定证据不足
        if not compressed_context:
            agent_log.append("🧠 推理: 上下文为空，证据不足 → 放弃重检索")
            return {
                "reasoning_draft": "（上下文为空，无法进行推理）",
                "need_reretrieve": False,
                "re_retrieve_queries": [],
                "agent_log": agent_log,
            }

        agent_log.append(f"🧠 CoT 推理中... (上下文 ~{len(compressed_context)} 字符)")

        logger.info(
            "[ReasonAgent.CoT] question=%s context_len=%d conflict=%s",
            question[:80], len(compressed_context), bool(conflict_note),
        )

        try:
            chain = COT_REASON_PROMPT | self._reason_llm
            response = await chain.ainvoke({
                "question": question,
                "context": compressed_context,
                "conflict_note": conflict_note or "（无冲突）",
            })
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            reasoning_draft: str = result.get("reasoning_draft", "")
            need_reretrieve: bool = bool(result.get("need_reretrieve", False))
            re_retrieve_queries: list = result.get("re_retrieve_queries", [])

        except Exception as e:
            logger.warning("[ReasonAgent.CoT] LLM/parse failed (%s), fallback.", e)
            reasoning_draft = (
                f"（推理异常: {e}）\n"
                f"基于上下文直接回答: {question}"
            )
            need_reretrieve = False
            re_retrieve_queries = []

        # 日志
        draft_preview = reasoning_draft[:100].replace("\n", " ")
        agent_log.append(f"🧠 推理完成: {len(reasoning_draft)} 字符 → {draft_preview}...")

        if need_reretrieve:
            agent_log.append(
                f"🔁 证据不足，生成 {len(re_retrieve_queries)} 个补充查询"
            )
            for i, sq in enumerate(re_retrieve_queries):
                agent_log.append(f"   RQ{i+1}: {sq['query'][:80]}")
        else:
            agent_log.append("✅ 证据充足，进入答案生成")

        logger.info(
            "[ReasonAgent.CoT] draft=%d chars reretrieve=%s queries=%d",
            len(reasoning_draft), need_reretrieve, len(re_retrieve_queries),
        )

        return {
            "reasoning_draft": reasoning_draft,
            "need_reretrieve": need_reretrieve,
            "re_retrieve_queries": re_retrieve_queries,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 设置路由 ----

    async def finalize(self, state: ReasonState) -> Dict[str, Any]:
        """根据 need_reretrieve 决定下游路由。"""
        need_reretrieve = state.get("need_reretrieve", False)
        agent_log: list[str] = list(state.get("agent_log", []))

        if need_reretrieve:
            route_action = "retriever_agent"
            agent_log.append("🔁 路由: → retriever_agent（二次检索）")
        else:
            route_action = "writer_agent"
            agent_log.append("➡️ 路由: → writer_agent（答案生成）")

        return {
            "route_action": route_action,
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_reason_agent():
    """构建并编译 ReasonAgent 子图。

    拓扑：
        START → cot_reason → finalize → END
    """
    agent = ReasonAgent()

    workflow = StateGraph(ReasonState)

    workflow.add_node("cot_reason", agent.cot_reason)
    workflow.add_node("finalize", agent.finalize)

    workflow.set_entry_point("cot_reason")
    workflow.add_edge("cot_reason", "finalize")
    workflow.add_edge("finalize", END)

    compiled = workflow.compile()

    logger.info(
        "ReasonAgent subgraph compiled. "
        "Topology: START → cot_reason → finalize → END"
    )

    return compiled
