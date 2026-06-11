"""WriterAgent — 答案生成节点。

职责：
1. 读取推理草稿 + 压缩上下文，生成完整回答 raw_answer
2. 自动添加文档引用标注 [doc_xxx]，脱敏内部敏感编号
3. 按业务风格格式化（列表 / 表格 / 段落）

内部工作流（2 节点）：

    START
      │
      ▼
   generate ─→ LLM 生成答案（含引用 + 脱敏 + 格式化）
      │
      ▼
   parse_sources ─→ 提取引用 doc_id → 构建 sources 列表
      │
      ▼
     END

下游固定节点：anti_hallucination_agent
"""

from __future__ import annotations

import re
from typing import Dict, Any, List

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.core.llm_factory import get_llm
from app.graph.agents.writer.state import WriterState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Prompt
# ============================================================================

GENERATE_ANSWER_PROMPT = ChatPromptTemplate.from_template("""你是一个企业级智能助手，需要基于以下信息生成专业、准确的回答。

<用户问题>
{question}
</用户问题>

<推理框架>
{reasoning_draft}
</推理框架>

<参考上下文>
{context}
</参考上下文>

## 生成规则

### 1. 内容准确性
- **严格基于上下文和推理框架**：每个关键事实必须能追溯到 <doc_id> 来源
- 如果上下文中没有足够信息，诚实说明 "根据现有知识库，暂未找到相关信息"
- 严禁编造事实、数字、日期

### 2. 引用标注
- 每个关键事实陈述后，必须用方括号标注来源文档 ID
- 格式：[doc_xxx] 或 [doc_xxx][doc_yyy]（多个来源）
- ID 必须与上下文中的 <doc_id> 完全一致
- 示例："2024年Q3营收为 1200 万元[doc_abc123]，同比增长 15%[doc_def456]"

### 3. 敏感信息脱敏
- 电话号码：替换为 "****" 或格式说明（如 "010-****-****"）
- 身份证号：完全隐藏，替换为 "***"
- 内部项目代号（如 "PJ-XXXX-001"）：替换为 "内部项目"
- 个人姓名（非公众人物）：替换为 "相关责任人"
- 银行账号 / 薪资数字：替换为 "****"

### 4. 格式化
- 对比类问题 → 使用表格
- 步骤/流程类 → 使用有序列表
- 多维度说明 → 使用无序列表 + 小标题
- 简单事实 → 清晰段落
- 使用与用户提问相同的语言

## 输出

请**直接输出最终回答**，不要包含任何前言、后记或解释。回答:""")


# ============================================================================
# 敏感信息脱敏正则
# ============================================================================

# 编译正则（作为 LLM 脱敏的补充，硬规则后处理）
_SENSITIVE_PATTERNS = [
    (re.compile(r'\b1[3-9]\d{9}\b'), '***'),                    # 手机号
    (re.compile(r'\b\d{3}-\d{4}-\d{4}\b'), '***-****-****'),    # 座机号
    (re.compile(r'\b\d{17}[\dXx]\b'), '***'),                    # 身份证
    (re.compile(r'\b\d{16,19}\b'), '***'),                       # 银行卡号
]


def _desensitize_hard(text: str) -> str:
    """硬规则脱敏 — 对 LLM 可能遗漏的敏感数字做兜底处理。"""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ============================================================================
# WriterAgent 类
# ============================================================================

class WriterAgent:
    """WriterAgent — 答案生成专家。

    基于推理草稿和上下文生成带引用标注的专业回答。
    """

    def __init__(self) -> None:
        self._writer_llm = get_llm(temperature=0.2, max_tokens=2048, timeout=45)

    # ---- 节点 1: 生成答案 ----

    async def generate(self, state: WriterState) -> Dict[str, Any]:
        """LLM 生成答案（含引用 + 脱敏 + 格式化）。"""
        question = state.get("question", "")
        compressed_context = state.get("compressed_context", "")
        reasoning_draft = state.get("reasoning_draft", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        # 空上下文 → 直接返回无法回答
        if not compressed_context:
            raw_answer = "抱歉，根据现有知识库，暂未找到与您问题相关的信息。请尝试换个方式提问，或联系管理员补充相关文档。"
            agent_log.append("✍️ 生成: 上下文为空，返回默认响应")
            prev_details = state.get("retrieval_details", {})
            return {
                "raw_answer": raw_answer,
                "sources": [],
                "retrieval_details": {
                    "doc_count": 0,
                    "rerank_scores": prev_details.get("rerank_scores", []),
                },
                "agent_log": agent_log,
            }

        agent_log.append(f"✍️ 生成答案中... (上下文 ~{len(compressed_context)} 字符)")

        logger.info(
            "[WriterAgent.Generate] question=%s context_len=%d draft_len=%d",
            question[:80], len(compressed_context), len(reasoning_draft),
        )

        try:
            chain = GENERATE_ANSWER_PROMPT | self._writer_llm
            response = await chain.ainvoke({
                "question": question,
                "reasoning_draft": reasoning_draft or "（无推理草稿，请直接基于上下文回答）",
                "context": compressed_context,
            })
            raw_answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
        except Exception as e:
            logger.error("[WriterAgent.Generate] LLM failed (%s), fallback.", e)
            raw_answer = "抱歉，答案生成服务暂时不可用，请稍后重试。"
            agent_log.append(f"⚠️ 生成失败: {e}")

        # 硬规则兜底脱敏
        raw_answer = _desensitize_hard(raw_answer)

        agent_log.append(f"✍️ 生成完成: {len(raw_answer)} 字符")

        logger.info("[WriterAgent.Generate] %d chars generated", len(raw_answer))

        return {
            "raw_answer": raw_answer,
            "_context": compressed_context,  # 传给 parse_sources 用于引用匹配
            "agent_log": agent_log,
        }

    # ---- 节点 2: 引用解析 + 格式化输出 ----

    async def parse_sources(self, state: WriterState) -> Dict[str, Any]:
        """提取 [doc_xxx] 引用，匹配上下文中的来源信息。"""
        raw_answer: str = state.get("raw_answer", "")
        compressed_context: str = state.get("_context", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        # 从 raw_answer 中提取所有 [doc_xxx] 引用
        cited_ids: List[str] = list(set(
            re.findall(r'\[(doc_[a-zA-Z0-9_]+)\]', raw_answer)
        ))

        # 从上下文中提取 doc_id → source_file + page 的映射
        source_map: Dict[str, dict] = {}
        if compressed_context:
            # 匹配 <doc_id>xxx</doc_id>
            id_matches = re.findall(
                r'<doc_id>(doc_[a-zA-Z0-9_]+)</doc_id>', compressed_context
            )
            for doc_id in id_matches:
                if doc_id not in source_map:
                    # 尝试提取同一 doc 块中的 source_file 和 page
                    block_pattern = re.compile(
                        rf'<doc_id>{re.escape(doc_id)}</doc_id>.*?'
                        rf'<source_file>(.*?)</source_file>.*?'
                        rf'<page>(.*?)</page>',
                        re.DOTALL,
                    )
                    block_match = block_pattern.search(compressed_context)
                    if block_match:
                        source_map[doc_id] = {
                            "id": doc_id,
                            "source_file": block_match.group(1).strip(),
                            "page": block_match.group(2).strip(),
                            "type": "Text",
                            "snippet": "",
                        }

        # 构建 sources
        sources: List[Dict[str, Any]] = []
        for cid in cited_ids:
            if cid in source_map:
                sources.append(source_map[cid])
            else:
                # 引用 ID 在上下文中找不到 → 可能是 LLM 编造的
                logger.warning(
                    "[WriterAgent.Parse] Unknown citation: %s", cid
                )
                sources.append({
                    "id": cid,
                    "source_file": "Unknown",
                    "page": "N/A",
                    "type": "Unknown",
                    "snippet": "（引用未在上下文中找到）",
                })

        if not sources and raw_answer:
            agent_log.append("⚠️ 引用解析: 未检测到 [doc_xxx] 引用标记")
        else:
            agent_log.append(
                f"📎 引用解析: {len(cited_ids)} 个引用 → "
                f"{len(sources)} 个有效来源"
            )

        # 统计 doc_count，并保留 retriever 传入的 rerank_scores
        prev_details = state.get("retrieval_details", {})
        doc_count = len(set(s.get("id") for s in sources))
        rerank_scores = prev_details.get("rerank_scores", [])

        logger.info(
            "[WriterAgent.Parse] %d citations → %d unique sources",
            len(cited_ids), doc_count,
        )

        return {
            "sources": sources,
            "retrieval_details": {
                "doc_count": doc_count,
                "rerank_scores": rerank_scores,
            },
            "route_action": "anti_hallucination_agent",
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_writer_agent():
    """构建并编译 WriterAgent 子图。

    拓扑：
        START → generate → parse_sources → END
    """
    agent = WriterAgent()

    workflow = StateGraph(WriterState)

    workflow.add_node("generate", agent.generate)
    workflow.add_node("parse_sources", agent.parse_sources)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "parse_sources")
    workflow.add_edge("parse_sources", END)

    compiled = workflow.compile()
    return compiled
