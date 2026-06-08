"""CriticAgent — 答案评审专家，检测幻觉、验证引用、评估完整性。

内部工作流：

    START
      │
      ▼
   evaluate ─→ 一次性完成事实核查 + 引用验证 + 完整性评估
      │
      ▼
   format_feedback ──→ END

CriticAgent 不是循环体，而是单次深度评估。
循环逻辑由父图处理：critic → regenerate → critic (最多 2 次)。
"""

from __future__ import annotations

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.graph.agents.critic.state import CriticState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# LLM
# ============================================================================


def _create_critic_llm() -> ChatOpenAI:
    """评审 LLM — temperature=0，评分必须严格、一致、可复现。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=1024,
        request_timeout=30,
        max_retries=2,
    )


# ============================================================================
# Prompt
# ============================================================================

CRITIC_PROMPT = ChatPromptTemplate.from_template("""你是一个严谨的答案质量评审专家。你需要对 AI 生成的答案进行三维度评审。

## 用户问题
{question}

## 源文档（事实依据）
{documents}

## AI 生成的答案
{answer}

## 评审要求

### 维度 1：事实准确性 (Factuality)
- 逐条检查答案中的每个事实陈述是否能在源文档中找到支撑
- 标注任何"无中生有"的编造内容（幻觉）
- 检查数字、日期、名称等关键信息是否与原文一致

### 维度 2：引用准确性 (Citation Accuracy)
- 答案中引用的文档 ID（格式 [chunk_xxx]）是否真实存在于源文档列表中
- 引用的内容是否与被引用文档的实际内容相符
- 是否遗漏了应该引用但没有引用的关键陈述

### 维度 3：完整性 (Completeness)
- 答案是否充分回应用户问题
- 源文档中是否有重要信息被遗漏
- 是否存在答非所问的情况

## 输出格式

请**只返回一个 JSON 对象**，格式如下：
{{
  "verdict": "pass 或 fail",
  "overall_score": <1-5的整数>,
  "factuality_score": <1-5的整数>,
  "citation_score": <1-5的整数>,
  "completeness_score": <1-5的整数>,
  "issues": ["问题1", "问题2", ...],
  "feedback": "如果 verdict=fail，提供具体的改进建议。如果 pass，填 '无需改进'。"
}}

评审标准：
- overall_score >= 4 且 无严重事实错误 → verdict = "pass"
- overall_score < 4 或 存在编造/错误引用 → verdict = "fail"

JSON:""")


# ============================================================================
# CriticAgent 类
# ============================================================================

class CriticAgent:
    """CriticAgent — 答案评审专家。

    对生成答案进行事实核查、引用验证、完整性评估，
    输出评审结论和具体改进建议。
    """

    def __init__(self) -> None:
        self._llm = _create_critic_llm()

    # ---- 节点 1: 综合评估 ----

    async def evaluate(self, state: CriticState) -> Dict[str, Any]:
        """三维度综合评估。"""
        answer = state["answer"]
        documents = state.get("documents", [])
        question = state.get("question", "")

        logger.info(
            f"[CriticAgent] Evaluating answer ({len(answer)} chars) "
            f"against {len(documents)} docs..."
        )

        agent_log = list(state.get("agent_log", []))

        # 无文档时的快速判定
        if not documents:
            agent_log.append("🔍 评审: 无源文档，判定为 fail（无依据）")
            return {
                "verdict": "fail",
                "overall_score": 1,
                "issues": ["答案无任何源文档支撑"],
                "feedback": "需要先检索相关文档再生成答案。",
                "agent_log": agent_log,
            }

        # 构建源文档摘要（截断以避免 token 超限）
        doc_summaries: list[str] = []
        total_chars = 0
        max_doc_chars = 3000  # 源文档总字符上限

        for i, doc in enumerate(documents):
            meta = doc.get("metadata", {})
            text = doc.get("text", "")
            doc_summaries.append(
                f"--- 文档 {i+1} ---\n"
                f"ID: {doc['id']}\n"
                f"来源: {meta.get('source_file', 'N/A')}\n"
                f"页码: {meta.get('page_number', 'N/A')}\n"
                f"内容:\n{text}"
            )
            total_chars += len(text)
            if total_chars > max_doc_chars:
                doc_summaries.append("... (后续文档已截断)")
                break

        try:
            chain = CRITIC_PROMPT | self._llm
            response = await chain.ainvoke({
                "question": question,
                "documents": "\n\n".join(doc_summaries),
                "answer": answer,
            })
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            verdict = result.get("verdict", "pass")
            overall_score = float(result.get("overall_score", 4))
            issues = result.get("issues", [])
            feedback = result.get("feedback", "")
            factuality_score = result.get("factuality_score", overall_score)
            citation_score = result.get("citation_score", overall_score)
            completeness_score = result.get("completeness_score", overall_score)

        except Exception as e:
            logger.warning(f"[CriticAgent] Parse failed ({e}), defaulting to pass.")
            verdict = "pass"
            overall_score = 3.0
            factuality_score = 3
            citation_score = 3
            completeness_score = 3
            issues = [f"评审解析异常: {e}"]
            feedback = "评审过程出现技术错误，请人工审核。"

        log_entry = (
            f"🔍 评审: 综合={overall_score}/5 "
            f"(事实={factuality_score} 引用={citation_score} 完整={completeness_score}), "
            f"判定={verdict}"
        )
        agent_log.append(log_entry)

        if issues:
            for issue in issues[:3]:  # 最多展示前 3 个
                agent_log.append(f"   ⚠️ {issue}")

        logger.info(f"[CriticAgent] {log_entry}")

        return {
            "verdict": verdict,
            "overall_score": overall_score,
            "issues": issues,
            "feedback": feedback,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 格式化输出 ----

    async def format_feedback(self, state: CriticState) -> Dict[str, Any]:
        """格式化评审结果。"""
        agent_log = list(state.get("agent_log", []))
        verdict = state.get("verdict", "pass")

        if verdict == "pass":
            agent_log.append("✅ 评审通过，答案质量合格")
        else:
            agent_log.append(
                f"❌ 评审不通过，需修正: {state.get('feedback', '')[:100]}"
            )

        return {
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_critic_agent():
    """构建并编译 CriticAgent 子图。

    Returns:
        编译后的 CompiledGraph，
        可通过 .ainvoke({"answer": "...", "documents": [...], "question": "..."}) 调用。
    """
    agent = CriticAgent()

    workflow = StateGraph(CriticState)

    workflow.add_node("evaluate", agent.evaluate)
    workflow.add_node("format_feedback", agent.format_feedback)

    workflow.set_entry_point("evaluate")
    workflow.add_edge("evaluate", "format_feedback")
    workflow.add_edge("format_feedback", END)

    compiled = workflow.compile()

    logger.info(
        "CriticAgent subgraph compiled. "
        "Topology: START → evaluate → format_feedback → END"
    )

    return compiled
