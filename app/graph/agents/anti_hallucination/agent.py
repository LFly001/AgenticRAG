"""AntiHallucinationAgent — 幻觉检测校验节点（终端节点）。

职责：
1. 逐句比对 raw_answer 与 valid_docs 原始知识库内容
2. 识别无依据编造、篡改原文、虚假数据
3. 修正错误语句，生成最终可信答案 final_answer
4. 标记幻觉风险等级 hallucination_risk（none / mild / high）

内部工作流（2 节点）：

    START
      │
      ▼
   verify_and_correct ─→ LLM 逐句核查 + 修正 + 风险评级
      │
      ▼
   finalize ─→ 设置 route_action="end"，流程终止
      │
      ▼
     END

路由：流程终止，返回 final_answer 给用户。
"""

from __future__ import annotations

import json
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.graph.agents.anti_hallucination.state import AntiHallucinationState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# LLM
# ============================================================================

def _create_hallucination_llm() -> ChatOpenAI:
    """幻觉检测 LLM — temperature=0，评审必须严格一致。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=2048,
        request_timeout=45,
        max_retries=2,
    )


# ============================================================================
# Prompt
# ============================================================================

VERIFY_AND_CORRECT_PROMPT = ChatPromptTemplate.from_template("""你是一个严格的答案质量审核专家。请逐句核查以下 AI 生成的答案，比对原始知识库文档，识别并修正任何幻觉（编造、篡改、虚假数据）。

<用户问题>
{question}
</用户问题>

<AI 生成的答案>
{raw_answer}
</AI 生成的答案>

<原始知识库文档（事实依据）>
{valid_docs}
</原始知识库文档>

## 核查步骤

### 步骤 1：逐句比对
将答案拆分为独立的陈述句，逐句检查：
1. **有依据**：该陈述在原始文档中能找到对应内容 → 保留
2. **无依据编造**：陈述的内容在文档中完全找不到 → 删除或标记
3. **篡改原文**：引用了文档但改变了关键数字/日期/名称 → 修正为原文准确值
4. **虚假数据**：数字、百分比、金额等在文档中不存在或不同 → 修正或删除
5. **引用错误**：[doc_xxx] 指向的文档与陈述内容不匹配 → 修正引用

### 步骤 2：修正错误
- 对于可修正的错误：直接替换为文档中的准确内容
- 对于无法修正的编造：删除该陈述
- 如果删除后影响上下文连贯性，用平滑过渡

### 步骤 3：风险评级
根据发现的问题严重程度评级：
- **none**（无风险）：所有陈述都有文档依据，无任何编造或错误
- **mild**（轻度）：存在措辞不够精确、引用格式不规范等小问题，但不影响事实准确性
- **high**（高风险）：存在明显编造、虚假数字、篡改关键事实等严重问题

## 输出格式

请**只返回一个 JSON 对象**：

{{
  "final_answer": "修正后的最终答案文本...",
  "hallucination_risk": "none",
  "issues_found": ["发现的问题1", "问题2"],
  "corrections_made": ["修正1", "修正2"]
}}

JSON:""")


# ============================================================================
# AntiHallucinationAgent 类
# ============================================================================

class AntiHallucinationAgent:
    """AntiHallucinationAgent — 幻觉检测校验专家。

    作为 8-Agent 管道的终端节点，负责最后的质量把关：
    逐句核查 → 修正错误 → 输出可信最终答案。
    """

    def __init__(self) -> None:
        self._verify_llm = _create_hallucination_llm()

    # ---- 节点 1: 核查 + 修正 ----

    async def verify_and_correct(self, state: AntiHallucinationState) -> Dict[str, Any]:
        """LLM 逐句核查 raw_answer，修正幻觉，输出 final_answer。"""
        question = state.get("question", "")
        raw_answer = state.get("raw_answer", "")
        valid_docs: List[Dict[str, Any]] = state.get("valid_docs", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        # 无 valid_docs → 无法核查，raw_answer 即为 final_answer，标记高风险
        if not valid_docs:
            agent_log.append("🛡️ 核查: 无 valid_docs，标记高风险")
            return {
                "final_answer": raw_answer or "（无法生成有效答案）",
                "hallucination_risk": "high",
                "_issues_found": ["无可信文档进行核查"],
                "_corrections_made": [],
                "agent_log": agent_log,
            }

        # 无 raw_answer → 直接返回
        if not raw_answer:
            agent_log.append("🛡️ 核查: raw_answer 为空，跳过")
            return {
                "final_answer": "抱歉，未能生成有效答案。",
                "hallucination_risk": "none",
                "_issues_found": [],
                "_corrections_made": [],
                "agent_log": agent_log,
            }

        agent_log.append(f"🛡️ 幻觉检测中... (答案 {len(raw_answer)} 字符, {len(valid_docs)} 篇文档)")

        logger.info(
            "[AntiHallucination.Verify] answer=%d chars docs=%d",
            len(raw_answer), len(valid_docs),
        )

        # 构建文档摘要（截断以控制 token）
        doc_blocks: list[str] = []
        total_chars = 0
        max_doc_chars = 3500

        for i, doc in enumerate(valid_docs):
            meta = doc.get("metadata", {}) or {}
            text = (doc.get("text", "") or "").strip()
            block = (
                f"--- 文档 {i+1} (ID: {doc.get('id', '?')}, "
                f"来源: {meta.get('source_file', '?')}) ---\n{text}"
            )
            doc_blocks.append(block)
            total_chars += len(text)
            if total_chars > max_doc_chars:
                doc_blocks.append("... (后续文档已截断)")
                break

        valid_docs_text = "\n\n".join(doc_blocks)

        try:
            chain = VERIFY_AND_CORRECT_PROMPT | self._verify_llm
            response = await chain.ainvoke({
                "question": question,
                "raw_answer": raw_answer,
                "valid_docs": valid_docs_text,
            })
            result_raw = response.content if hasattr(response, "content") else str(response)
            result_raw = result_raw.strip()
            if result_raw.startswith("```"):
                result_raw = result_raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(result_raw)
            final_answer: str = result.get("final_answer", raw_answer)
            hallucination_risk: str = result.get("hallucination_risk", "none")
            issues_found: list = result.get("issues_found", [])
            corrections_made: list = result.get("corrections_made", [])

        except Exception as e:
            logger.warning(
                "[AntiHallucination.Verify] LLM/parse failed (%s), passthrough.", e
            )
            final_answer = raw_answer
            hallucination_risk = "mild"
            issues_found = [f"核查异常: {e}"]
            corrections_made = []

        # 校验风险等级
        if hallucination_risk not in ("none", "mild", "high"):
            hallucination_risk = "mild"

        # 日志
        risk_labels = {"none": "🟢 无风险", "mild": "🟡 轻度风险", "high": "🔴 高风险"}
        agent_log.append(
            f"🛡️ 核查完成: {risk_labels.get(hallucination_risk, hallucination_risk)}"
        )

        if issues_found:
            for issue in issues_found[:3]:
                agent_log.append(f"   ⚠️ {issue}")
        if corrections_made:
            for corr in corrections_made[:3]:
                agent_log.append(f"   ✅ 修正: {corr}")

        logger.info(
            "[AntiHallucination.Verify] risk=%s issues=%d corrections=%d",
            hallucination_risk, len(issues_found), len(corrections_made),
        )

        return {
            "final_answer": final_answer,
            "hallucination_risk": hallucination_risk,
            "_issues_found": issues_found,
            "_corrections_made": corrections_made,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 终止 ----

    async def finalize(self, state: AntiHallucinationState) -> Dict[str, Any]:
        """设置流程终止标记。"""
        agent_log: list[str] = list(state.get("agent_log", []))
        agent_log.append("🏁 流程终止: final_answer 已就绪")

        return {
            "route_action": "end",
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_anti_hallucination_agent():
    """构建并编译 AntiHallucinationAgent 子图（终端节点）。

    拓扑：
        START → verify_and_correct → finalize → END
    """
    agent = AntiHallucinationAgent()

    workflow = StateGraph(AntiHallucinationState)

    workflow.add_node("verify_and_correct", agent.verify_and_correct)
    workflow.add_node("finalize", agent.finalize)

    workflow.set_entry_point("verify_and_correct")
    workflow.add_edge("verify_and_correct", "finalize")
    workflow.add_edge("finalize", END)

    compiled = workflow.compile()

    logger.info(
        "AntiHallucinationAgent subgraph compiled (terminal). "
        "Topology: START → verify_and_correct → finalize → END"
    )

    return compiled
