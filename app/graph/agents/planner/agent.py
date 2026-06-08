"""QueryPlanner — 查询规划专家，判断复杂度并拆解复合问题为子查询序列。

内部工作流：

    START
      │
      ▼
   analyze_complexity ─→ 判断: simple | complex
      │
      ├── simple ──────────→ skip_decompose ──→ END
      │
      └── complex ─→ decompose ─→ format_plan ──→ END

子查询格式：每个子查询包含 query、strategy、priority 三个字段。
父图根据 is_complex 决定走单检索还是并行多检索。
"""

from __future__ import annotations

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.graph.agents.planner.state import PlannerState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# LLM
# ============================================================================


def _create_planner_llm() -> ChatOpenAI:
    """规划 LLM — temperature=0，拆解必须稳定一致。"""
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
# Prompts
# ============================================================================

COMPLEXITY_PROMPT = ChatPromptTemplate.from_template("""你是一个查询复杂度分析专家。判断用户问题是否属于"复合问题"。

**简单问题**（不需要拆解）：
- 单一事实查询："2024年Q3营收是多少？"
- 单一概念解释："什么是OKR？"
- 单一流程："请假流程是什么？"

**复合问题**（需要拆解为子问题）：
- 多实体比较："对比A产品和B产品的性能参数"
- 多时间段："2023年和2024年的研发投入变化"
- 多维度："公司的人力资源政策包括哪些方面？分别是什么？"
- 条件分支："如果销售额超过100万，提成比例是多少？否则呢？"
- 多跳推理："谁是我们最畅销产品的负责人？他的联系方式是什么？"

Question: {question}

请**只返回一个 JSON 对象**，格式如下：
{{"is_complex": true或false, "reason": "<一句话说明判断原因>"}}

JSON:""")


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template("""你是一个查询拆解专家。将复合问题拆解为可独立检索的子问题。

原始问题: {question}
复杂度原因: {reason}

拆解规则：
1. 每个子问题必须能独立检索
2. 子问题之间应有清晰的逻辑边界
3. 为每个子问题标注推荐策略（vector/bm25/hybrid）
4. 标注优先级（1=最高），数字越小越优先
5. 子问题数量不超过 5 个

请**只返回一个 JSON 数组**，格式如下：
[
  {{"query": "子问题1文本", "strategy": "hybrid", "priority": 1}},
  {{"query": "子问题2文本", "strategy": "bm25", "priority": 2}}
]

JSON:""")


# ============================================================================
# QueryPlanner 类
# ============================================================================

class QueryPlanner:
    """QueryPlanner — 查询规划专家。

    分析问题复杂度，将复合问题拆解为子查询序列，
    为父图的并行检索提供调度依据。
    """

    def __init__(self) -> None:
        self._llm = _create_planner_llm()

    # ---- 节点 1: 复杂度分析 ----

    async def analyze_complexity(self, state: PlannerState) -> Dict[str, Any]:
        """分析问题是否需要拆解。"""
        question = state["question"]
        logger.info(f"[Planner.Analyze] {question[:80]}...")

        agent_log: list[str] = []

        try:
            chain = COMPLEXITY_PROMPT | self._llm
            response = await chain.ainvoke({"question": question})
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            is_complex = bool(result.get("is_complex", False))
            reason = result.get("reason", "N/A")
        except Exception as e:
            logger.warning(f"[Planner.Analyze] Parse failed ({e}), defaulting to simple.")
            is_complex = False
            reason = f"Parser error, default: {e}"

        log_entry = f"📋 复杂度分析: {'复合' if is_complex else '简单'}（{reason}）"
        agent_log.append(log_entry)
        logger.info(f"[Planner.Analyze] {log_entry}")

        return {
            "is_complex": is_complex,
            "complexity_reason": reason,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 问题拆解 ----

    async def decompose(self, state: PlannerState) -> Dict[str, Any]:
        """将复合问题拆解为子查询列表。"""
        question = state["question"]
        reason = state.get("complexity_reason", "")
        agent_log = list(state.get("agent_log", []))

        logger.info(f"[Planner.Decompose] Breaking down: {question[:80]}...")

        try:
            chain = DECOMPOSE_PROMPT | self._llm
            response = await chain.ainvoke({
                "question": question,
                "reason": reason,
            })
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            sub_queries = json.loads(raw)
            if not isinstance(sub_queries, list):
                sub_queries = [sub_queries]  # 容错：单对象转列表
        except Exception as e:
            logger.warning(f"[Planner.Decompose] Parse failed ({e}), using original question.")
            sub_queries = [{"query": question, "strategy": "hybrid", "priority": 1}]

        # 验证并补全子查询字段
        for sq in sub_queries:
            if "strategy" not in sq:
                sq["strategy"] = "hybrid"
            if "priority" not in sq:
                sq["priority"] = 1

        # 按优先级排序
        sub_queries.sort(key=lambda x: int(x.get("priority", 1)))

        log_entry = f"🔨 拆解结果: {len(sub_queries)} 个子问题"
        agent_log.append(log_entry)
        for i, sq in enumerate(sub_queries):
            agent_log.append(f"   Q{i+1}[{sq['strategy']}]: {sq['query'][:80]}")

        logger.info(f"[Planner.Decompose] {log_entry}")
        for sq in sub_queries:
            logger.info(f"  - [{sq['strategy']}] {sq['query'][:100]}")

        return {
            "sub_queries": sub_queries,
            "agent_log": agent_log,
        }

    # ---- 节点 3: 跳过拆解（简单问题） ----

    async def skip_decompose(self, state: PlannerState) -> Dict[str, Any]:
        """简单问题：用原始查询构造单个子查询。"""
        question = state["question"]
        agent_log = list(state.get("agent_log", []))
        agent_log.append("⏭️ 跳过拆解，直接检索")

        return {
            "sub_queries": [{"query": question, "strategy": "hybrid", "priority": 1}],
            "agent_log": agent_log,
        }

    # ---- 节点 4: 格式化输出 ----

    async def format_plan(self, state: PlannerState) -> Dict[str, Any]:
        """格式化规划结果。"""
        sub_queries = state.get("sub_queries", [])
        is_complex = state.get("is_complex", False)
        agent_log = list(state.get("agent_log", []))

        if is_complex:
            agent_log.append(f"✅ 规划完成: {len(sub_queries)} 个子查询，将并行检索")
        else:
            agent_log.append("✅ 简单查询，直接检索")

        return {
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_planner_agent():
    """构建并编译 QueryPlanner 子图。"""
    agent = QueryPlanner()

    workflow = StateGraph(PlannerState)

    workflow.add_node("analyze_complexity", agent.analyze_complexity)
    workflow.add_node("decompose", agent.decompose)
    workflow.add_node("skip_decompose", agent.skip_decompose)
    workflow.add_node("format_plan", agent.format_plan)

    workflow.set_entry_point("analyze_complexity")

    # 条件分支：complex → decompose, simple → skip
    workflow.add_conditional_edges(
        "analyze_complexity", _route_after_analyze,
        {"decompose": "decompose", "skip_decompose": "skip_decompose"},
    )

    workflow.add_edge("decompose", "format_plan")
    workflow.add_edge("skip_decompose", "format_plan")
    workflow.add_edge("format_plan", END)

    compiled = workflow.compile()

    logger.info(
        "QueryPlanner subgraph compiled. "
        "Topology: START → analyze → [decompose|skip] → format → END"
    )

    return compiled


def _route_after_analyze(state: PlannerState) -> str:
    is_complex = state.get("is_complex", False)
    if is_complex:
        return "decompose"
    return "skip_decompose"
