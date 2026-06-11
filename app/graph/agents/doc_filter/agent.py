"""DocFilterAgent — 文档校验清洗节点。

职责：
1. 去重、过滤低相似度无关片段
2. 时效校验：剔除过期作废文档
3. 识别文档内容冲突，写入 conflict_note
4. 过滤无意义碎片，输出可信文档集合 valid_docs

内部工作流（2 节点）：

    START
      │
      ▼
   filter_docs ─→ 规则过滤：分数阈值 + 空片段 + 时效 + 文本去重
      │
      ▼
   detect_conflicts ─→ LLM 冲突检测：识别文档间事实矛盾 → conflict_note
      │
      ▼
     END

下游固定节点：context_compress_agent
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.core.llm_factory import get_llm
from app.graph.agents.doc_filter.state import DocFilterState
from app.utils.llm_utils import parse_llm_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# 过滤阈值常量
# ============================================================================

MIN_TEXT_LENGTH = 10          # 最小文本长度（字符），低于此值视为碎片
MIN_RRF_SCORE = 0.0           # 最低 RRF 分数阈值（仅 reranker 不可用时启用）

# 时效相关元数据键名（按优先级）
EXPIRY_DATE_KEYS = [
    "valid_until", "expiry_date", "expire_date", "end_date",
    "obsolete_date", "deprecated_date", "invalid_after",
]
PUBLISH_DATE_KEYS = [
    "publish_date", "pub_date", "created_date", "date",
    "effective_date", "start_date",
]


# ============================================================================
# Prompt
# ============================================================================

CONFLICT_DETECT_PROMPT = ChatPromptTemplate.from_template("""你是一个文档一致性校验专家。检查以下文档之间是否存在事实性矛盾或冲突。

<documents>
{documents}
</documents>

检查规则：
1. 对比各文档中的数字、日期、名称、状态等关键事实
2. 如果同一事实在不同文档中有不同甚至相反的表述，记录为冲突
3. 如果文档之间只是侧重点不同而无实质矛盾，不算冲突
4. 版本号不同（如 v1.0 vs v2.0）通常不是冲突，而是更新

请**只返回一个 JSON 对象**：
{{
  "has_conflict": true或false,
  "conflict_note": "冲突说明"
}}

如果 has_conflict=false，conflict_note 为空字符串 ""。
如果 has_conflict=true，conflict_note 用 1-2 句话说明具体冲突内容。

JSON:""")


# ============================================================================
# DocFilterAgent 类
# ============================================================================

class DocFilterAgent:
    """DocFilterAgent — 文档校验清洗专家。

    对 RetrieveAgent 输出的 raw_docs 进行三重过滤：
    1. 碎片过滤 — 去除空文本 / 过短片段
    2. 时效校验 — 去除已过期的作废文档
    3. 文本去重 — 去除内容高度重复的文档
    + LLM 冲突检测
    """

    def __init__(self) -> None:
        self._conflict_llm = get_llm(temperature=0, max_tokens=512, timeout=25)

    # ---- 节点 1: 规则过滤 ----

    async def filter_docs(self, state: DocFilterState) -> Dict[str, Any]:
        """四重规则过滤 → _filtered 内部字段。"""
        documents: List[Dict[str, Any]] = state.get("documents", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        total_in = len(documents)
        logger.info("[DocFilter.Filter] IN: %d docs", total_in)
        if total_in == 0:
            agent_log.append("🫧 过滤: 输入为空，跳过")
            logger.warning("[DocFilter.Filter] EMPTY input!")
            return {"_filtered": [], "agent_log": agent_log}

        removed_empty = 0
        removed_score = 0
        removed_expired = 0
        removed_dup = 0

        now = datetime.now(timezone.utc)
        stage1: List[Dict[str, Any]] = []

        # --- 第一轮：碎片 + 分数 ---
        for doc in documents:
            text = (doc.get("text", "") or "").strip()
            # 空文本 / 碎片
            if len(text) < MIN_TEXT_LENGTH:
                removed_empty += 1
                logger.info(
                    "[DocFilter] Drop short chunk id=%s len=%d text=%.50s",
                    doc.get("id", "?")[:40], len(text), text,
                )
                continue
            # 分数阈值（仅 reranker 不可用时用 RRF 兜底，0.0 基本不滤）
            if doc.get("rerank_score") is None:
                if doc.get("rrf_score", 0) < MIN_RRF_SCORE:
                    removed_score += 1
                    continue
            stage1.append(doc)

        # --- 第二轮：时效校验 ---
        stage2: List[Dict[str, Any]] = []
        for doc in stage1:
            meta = doc.get("metadata", {}) or {}
            if self._is_expired(meta, now):
                removed_expired += 1
                continue
            stage2.append(doc)

        # --- 第三轮：文本去重（前 200 字符 hash） ---
        seen_hashes: set[str] = set()
        stage3: List[Dict[str, Any]] = []
        for doc in stage2:
            text = (doc.get("text", "") or "").strip()
            h = hashlib.md5(text[:200].encode('utf-8')).hexdigest()
            if h in seen_hashes:
                removed_dup += 1
                continue
            seen_hashes.add(h)
            stage3.append(doc)

        valid_docs = stage3

        # 日志
        parts: list[str] = [f"🫧 过滤: {total_in} → {len(valid_docs)}"]
        if removed_empty:
            parts.append(f"碎片-{removed_empty}")
        if removed_score:
            parts.append(f"低分-{removed_score}")
        if removed_expired:
            parts.append(f"过期-{removed_expired}")
        if removed_dup:
            parts.append(f"去重-{removed_dup}")
        agent_log.append("  ".join(parts))

        logger.info(
            "[DocFilter.Filter] %d → %d (empty=%d score=%d expired=%d dup=%d)",
            total_in, len(valid_docs),
            removed_empty, removed_score, removed_expired, removed_dup,
        )

        return {
            "_filtered": valid_docs,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 冲突检测 ----

    async def detect_conflicts(self, state: DocFilterState) -> Dict[str, Any]:
        """LLM 检测文档间事实冲突 → conflict_note。"""
        valid_docs: List[Dict[str, Any]] = state.get("_filtered", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        # 只有 0-1 篇文档 → 无需冲突检测
        if len(valid_docs) <= 1:
            agent_log.append("🫧 冲突检测: 文档≤1，跳过")
            return {
                "valid_docs": valid_docs,
                "conflict_note": "",
                "route_action": "context_compress_agent",
                "agent_log": agent_log,
            }

        # 构建文档摘要
        doc_summaries: list[str] = []
        total_chars = 0
        max_chars = 2500

        for i, doc in enumerate(valid_docs):
            meta = doc.get("metadata", {}) or {}
            text = (doc.get("text", "") or "").strip()
            doc_summaries.append(
                f"--- 文档 {i+1} (ID: {doc.get('id', '?')}, "
                f"来源: {meta.get('source_file', '?')}) ---\n"
                f"{text[:400]}"
            )
            total_chars += len(text)
            if total_chars > max_chars:
                doc_summaries.append("... (后续文档已截断)")
                break

        # LLM 冲突检测
        conflict_note = ""
        try:
            chain = CONFLICT_DETECT_PROMPT | self._conflict_llm
            response = await chain.ainvoke({
                "documents": "\n\n".join(doc_summaries),
            })
            raw = response.content if hasattr(response, "content") else str(response)
            result = parse_llm_json(raw)
            has_conflict = bool(result.get("has_conflict", False))
            conflict_note = result.get("conflict_note", "")

            if has_conflict and conflict_note:
                agent_log.append(f"⚠️ 内容冲突: {conflict_note[:100]}")
            else:
                agent_log.append("✅ 冲突检测: 无矛盾")
        except Exception as e:
            logger.warning("[DocFilter.Conflict] LLM failed (%s), skip.", e)
            agent_log.append("🫧 冲突检测: LLM 异常，跳过")

        logger.info(
            "[DocFilter.Conflict] %d docs checked, conflict=%s",
            len(valid_docs), bool(conflict_note),
        )

        return {
            "valid_docs": valid_docs,
            "conflict_note": conflict_note,
            "route_action": "context_compress_agent",
            "agent_log": agent_log,
        }

    # ---- 时效判断 ----

    @staticmethod
    def _is_expired(meta: dict, now: datetime) -> bool:
        """检查文档元数据中的过期日期。

        支持多种日期字段名和日期格式。
        """
        # 1. 直接过期日期
        for key in EXPIRY_DATE_KEYS:
            val = meta.get(key)
            if val:
                dt = DocFilterAgent._parse_date(val)
                if dt and dt < now:
                    return True

        # 2. 发布日期间接判断（超过 N 年的旧文档）
        for key in PUBLISH_DATE_KEYS:
            val = meta.get(key)
            if val:
                dt = DocFilterAgent._parse_date(val)
                if dt:
                    age_days = (now - dt).days
                    # 超过 10 年的文档标记为潜在过期
                    if age_days > 3650:
                        return True
                break  # 只取第一个有效发布日期

        return False

    @staticmethod
    def _parse_date(val) -> Optional[datetime]:
        """解析多种日期格式为 UTC datetime。"""
        if isinstance(val, datetime):
            return val
        if not isinstance(val, str):
            return None
        val = val.strip()
        # 尝试常见格式
        formats = [
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
            "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y年%m月%d日",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None


# ============================================================================
# 子图构建
# ============================================================================

def build_doc_filter_agent():
    """构建并编译 DocFilterAgent 子图。

    拓扑：
        START → filter_docs → detect_conflicts → END
    """
    agent = DocFilterAgent()

    workflow = StateGraph(DocFilterState)

    workflow.add_node("filter_docs", agent.filter_docs)
    workflow.add_node("detect_conflicts", agent.detect_conflicts)

    workflow.set_entry_point("filter_docs")
    workflow.add_edge("filter_docs", "detect_conflicts")
    workflow.add_edge("detect_conflicts", END)

    compiled = workflow.compile()
    return compiled
