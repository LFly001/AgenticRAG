"""ContextCompressAgent — 上下文压缩节点。

职责：
1. 对 valid_docs 构建结构化上下文，保留原文证据引用
2. 合并重复信息，控制总 token 长度不超限
3. 拼接 compressed_context，附带每条文档来源标记

内部工作流（2 节点）：

    START
      │
      ▼
   format_context ─→ 结构化 XML 格式 + token 预估
      │
      ▼
   compress ─→ 若超预算 → LLM 压缩（保留引用 + 合并重复）；否则透传
      │
      ▼
     END

下游固定节点：reason_agent
"""

from __future__ import annotations

from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.graph.agents.context_compress.state import ContextCompressState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# 压缩配置
# ============================================================================

MAX_CONTEXT_TOKENS = 3500        # 压缩后上下文最大 token 数
TOKENS_PER_CHAR_CN = 0.67        # 中文字符 → token 估算系数（约 1.5 char/token）
TOKENS_PER_CHAR_EN = 0.25        # 英文字符 → token 估算系数（约 4 char/token）


# ============================================================================
# LLM
# ============================================================================

def _create_compress_llm() -> ChatOpenAI:
    """压缩 LLM — temperature=0，输出必须稳定可复现。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=2048,
        request_timeout=40,
        max_retries=2,
    )


# ============================================================================
# Prompt
# ============================================================================

COMPRESS_PROMPT = ChatPromptTemplate.from_template("""你是一个上下文压缩专家。请对以下检索到的文档进行智能压缩，在保留关键证据的前提下大幅缩减长度。

<用户问题>
{question}
</用户问题>

<原始上下文>
{raw_context}
</原始上下文>

压缩规则：
1. **保留证据**：每个文档的 <doc_id>、<source_file> 标签必须完整保留，不得删除或修改
2. **关键信息**：保留与用户问题最相关的数字、日期、名称、定义、流程步骤
3. **合并重复**：如果多个文档包含相同或高度相似的信息，合并为一条，并在来源中列出所有相关 doc_id
4. **去冗余**：删除开场白、过渡句、客套话、重复描述
5. **保留结构**：如果有列表、步骤、分类，保留其层级结构
6. **语言一致**：保持原始文档的语言
7. **长度控制**：压缩后的内容控制在原长度的 30-50% 左右

输出格式：保持与原上下文相同的 XML 结构，仅压缩 <content> 部分的内容：

<compressed_context>
<doc>
<doc_id>chunk_xxx</doc_id>
<source_file>文件来源</source_file>
<page>页码</page>
<content>（压缩后的内容，保留关键事实）</content>
</doc>
...
</compressed_context>

请**直接输出压缩后的 XML**，不要包含任何解释。""")

# ============================================================================
# Token 估算
# ============================================================================


def estimate_tokens(text: str) -> int:
    """简单字符级 token 估算。

    - 中文字符 ≈ 1.5 char/token  → factor 0.67
    - 英文/数字/标点 ≈ 4 char/token → factor 0.25
    - 按字符类型加权平均
    """
    if not text:
        return 0

    cn_chars = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
    other_chars = len(text) - cn_chars

    return int(cn_chars * TOKENS_PER_CHAR_CN + other_chars * TOKENS_PER_CHAR_EN)


# ============================================================================
# ContextCompressAgent 类
# ============================================================================

class ContextCompressAgent:
    """ContextCompressAgent — 上下文压缩专家。

    将 valid_docs 构建为结构化 XML 上下文，若超过 token 预算则通过 LLM
    压缩以保留关键证据 + 合并重复信息。
    """

    def __init__(self) -> None:
        self._compress_llm = _create_compress_llm()

    # ---- 节点 1: 格式化上下文 + token 预估 ----

    async def format_context(self, state: ContextCompressState) -> Dict[str, Any]:
        """将 valid_docs 格式化为结构化 XML，预估 token 数。"""
        valid_docs: List[Dict[str, Any]] = state.get("valid_docs", [])
        question = state.get("question", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        logger.info("[ContextCompress.Format] IN: %d docs", len(valid_docs))
        if not valid_docs:
            agent_log.append("🗜️ 格式化: 无文档，compressed_context 为空")
            logger.warning("[ContextCompress.Format] EMPTY input!")
            return {
                "_raw_context": "",
                "_token_estimate": 0,
                "compressed_context": "",
                "route_action": "reason_agent",
                "agent_log": agent_log,
            }

        parts: list[str] = []
        total_chars = 0

        for i, doc in enumerate(valid_docs):
            meta = doc.get("metadata", {}) or {}
            doc_id = doc.get("id", f"doc_{i}")
            source_file = meta.get("source_file", "Unknown")
            page = meta.get("page_number", meta.get("page", "N/A"))
            text = (doc.get("text", "") or "").strip()

            block = (
                f"<doc>\n"
                f"  <doc_id>{doc_id}</doc_id>\n"
                f"  <source_file>{source_file}</source_file>\n"
                f"  <page>{page}</page>\n"
                f"  <content>{text}</content>\n"
                f"</doc>"
            )
            parts.append(block)
            total_chars += len(text)

        raw_context = "\n\n".join(parts)
        token_est = estimate_tokens(raw_context)

        agent_log.append(
            f"🗜️ 格式化: {len(valid_docs)} 篇文档 → "
            f"{total_chars} 字符, ~{token_est} tokens"
        )

        logger.info(
            "[ContextCompress.Format] %d docs, %d chars, ~%d tokens",
            len(valid_docs), total_chars, token_est,
        )

        return {
            "_raw_context": raw_context,
            "_token_estimate": token_est,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 压缩（超预算时） ----

    async def compress(self, state: ContextCompressState) -> Dict[str, Any]:
        """若超过 token 预算则 LLM 压缩，否则透传。"""
        raw_context: str = state.get("_raw_context", "")
        token_est: int = state.get("_token_estimate", 0)
        question = state.get("question", "")
        valid_docs = state.get("valid_docs", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        # 空上下文
        if not raw_context:
            return {
                "compressed_context": "",
                "route_action": "reason_agent",
                "agent_log": agent_log,
            }

        # 未超预算 → 直接透传
        if token_est <= MAX_CONTEXT_TOKENS:
            agent_log.append(
                f"🗜️ 无需压缩: ~{token_est}/{MAX_CONTEXT_TOKENS} tokens，直接透传"
            )
            logger.info("[ContextCompress] Within budget (%d/%d), passthrough.",
                         token_est, MAX_CONTEXT_TOKENS)
            return {
                "compressed_context": raw_context,
                "route_action": "reason_agent",
                "agent_log": agent_log,
            }

        # 超预算 → LLM 压缩
        agent_log.append(
            f"🗜️ 触发压缩: ~{token_est}/{MAX_CONTEXT_TOKENS} tokens → LLM 压缩中..."
        )
        logger.info("[ContextCompress] Over budget (%d/%d), compressing...",
                     token_est, MAX_CONTEXT_TOKENS)

        try:
            chain = COMPRESS_PROMPT | self._compress_llm
            response = await chain.ainvoke({
                "question": question,
                "raw_context": raw_context,
            })
            compressed = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()

            # 清除可能的 markdown 包裹
            if compressed.startswith("```"):
                compressed = compressed.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        except Exception as e:
            logger.error("[ContextCompress] LLM failed (%s), fallback to truncation.", e)
            # 回退：按字符截断
            max_chars = int(MAX_CONTEXT_TOKENS * 2.0)  # 粗估
            compressed = raw_context[:max_chars] + "\n\n... (上下文已截断)"
            agent_log.append(f"⚠️ 压缩失败 ({e})，回退为截断")

        compressed_tokens = estimate_tokens(compressed)
        compression_ratio = (
            int((1 - compressed_tokens / max(token_est, 1)) * 100)
        )
        agent_log.append(
            f"🗜️ 压缩完成: ~{token_est} → ~{compressed_tokens} tokens "
            f"(压缩 {compression_ratio}%)"
        )

        logger.info(
            "[ContextCompress] ~%d → ~%d tokens (ratio=%d%%)",
            token_est, compressed_tokens, compression_ratio,
        )

        return {
            "compressed_context": compressed,
            "route_action": "reason_agent",
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_context_compress_agent():
    """构建并编译 ContextCompressAgent 子图。

    拓扑：
        START → format_context → compress → END
    """
    agent = ContextCompressAgent()

    workflow = StateGraph(ContextCompressState)

    workflow.add_node("format_context", agent.format_context)
    workflow.add_node("compress", agent.compress)

    workflow.set_entry_point("format_context")
    workflow.add_edge("format_context", "compress")
    workflow.add_edge("compress", END)

    compiled = workflow.compile()

    logger.info(
        "ContextCompressAgent subgraph compiled. "
        "Topology: START → format_context → compress → END"
    )

    return compiled
