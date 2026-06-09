"""IntentAgent — 意图解析节点。

职责：
0. 闲聊检测：最先执行，识别问候/感谢/告别等非业务对话 → 短路终止
1. 指代消解：结合 chat_history 将模糊代词补全为独立完整问题
2. 复杂问题拆解：将复合问题拆为多个可独立检索的子查询 → query_list
3. 澄清判断：检测问题是否存在信息缺失/歧义 → need_clarify + clarify_msg

内部工作流（4 节点）：

    START
      │
      ▼
   classify_chat ─→ 闲聊检测（一次 LLM 调用，短路入口）
      │
      ├── is_chat=True ──→ chat_response ──→ END（返回固定回复，跳过后续全链路）
      │
      └── is_chat=False → resolve_anaphora → analyze_and_decompose → END

路由分支（由父图 route_dispatcher 处理）：
    - is_chat=True       → 结束流程，返回固定闲聊回复
    - need_clarify=True  → 结束流程，返回澄清话术给用户
    - need_clarify=False → 下游 retriever_agent
"""

from __future__ import annotations

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.graph.agents.intent.state import IntentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# LLM
# ============================================================================

def _create_intent_llm(temperature: float = 0) -> ChatOpenAI:
    """意图解析 LLM — temperature=0，解析必须稳定一致。"""
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
        max_tokens=1024,
        request_timeout=30,
        max_retries=2,
    )


# ============================================================================
# Prompts
# ============================================================================

CHAT_DETECT_PROMPT = ChatPromptTemplate.from_template("""你是一个对话分流专家。判断用户输入是否属于"闲聊"。

**闲聊**（is_chat=true）：
- 问候："你好"、"hi"、"早上好"、"在吗"
- 感谢："谢谢"、"多谢"、"辛苦了"
- 告别："再见"、"拜拜"、"回头见"
- 无意义输入："哈哈"、"嗯嗯"、"哦哦"
- 测试性输入："test"、"测试"
- 与知识库查询无关的日常聊天

**非闲聊**（is_chat=false）：
- 任何包含具体问题、需要查询信息、涉及业务内容的输入
- 即使带有礼貌用语（如"请问一下公司的年假政策"），只要包含实质性查询内容，就不是闲聊

用户输入: {user_query}

请**只返回一个 JSON 对象**：
{{"is_chat": true或false}}

JSON:""")


ANAPHORA_PROMPT = ChatPromptTemplate.from_template("""你是一个对话上下文融合专家。用户可能使用了代词或省略了上文信息，请结合对话历史将用户问题还原为一个独立、完整的问题。

<对话历史>
{history}
</对话历史>

<用户当前输入>
{user_query}
</用户当前输入>

规则：
1. 将代词（"它""那个""这个""他""她""第二个""上面的"等）替换为历史中对应的具体实体名称
2. 补全被省略的主语、宾语、定语
3. 如果用户追问含糊（"详细说说""具体呢""为什么"），结合历史推断具体要说明什么
4. 如果用户输入已是一个独立完整的问题，原样返回
5. 保持原始问题的语言和语气
6. **只返回改写后的问题文本**，不要包含任何解释、引号或前缀

改写后的问题:""")


DECOMPOSE_AND_CLARIFY_PROMPT = ChatPromptTemplate.from_template("""你是一个查询分析专家。你需要同时完成两项任务：拆解复杂问题 + 判断信息完整性。

## 任务 1：拆解复杂问题

判断用户问题是否属于"复合问题"：

**简单问题**（不需要拆解，query_list 只包含原问题一项）：
- 单一事实查询："2024年Q3营收是多少？"
- 单一概念解释："什么是OKR？"
- 单一流程："请假流程是什么？"

**复合问题**（需要拆解为多个子查询）：
- 多实体比较："对比A产品和B产品的性能参数"
- 多时间段："2023年和2024年的研发投入变化"
- 多维度："公司的人力资源政策包括哪些方面？分别是什么？"
- 条件分支："如果销售额超过100万，提成比例是多少？如果没超过呢？"
- 多跳推理："谁是我们最畅销产品的负责人？他的联系方式是什么？"

拆解规则：
- 每个子查询必须能独立检索
- 子查询之间应有清晰的逻辑边界
- 子查询数量不超过 5 个

## 任务 2：判断信息完整性

检查用户问题是否存在以下情况：
- **关键实体缺失**：问题中提到"那个项目""我们部门"但未明确具体名称
- **条件不明确**："最近的销售额"——"最近"是指本周？本月？本季度？
- **指代无法消解**：即使用了历史也无法确定指代对象
- **问题过于模糊**：无法判断用户具体想问什么

如果存在上述任一情况，则需要向用户澄清。

## 用户问题
{resolved_query}

## 输出格式

请**只返回一个 JSON 对象**，格式如下：
{{
  "query_list": [
    {{"query": "子查询1文本"}},
    {{"query": "子查询2文本"}}
  ],
  "need_clarify": false,
  "clarify_msg": ""
}}

如果 need_clarify=true，clarify_msg 应是一句友好的追问，引导用户补充缺失信息。
如果 need_clarify=false，clarify_msg 留空字符串。

JSON:""")


# ============================================================================
# IntentAgent 类
# ============================================================================

class IntentAgent:
    """IntentAgent — 意图解析专家。

    内部三步流水线（闲聊优先短路）：
    0. classify_chat         — 闲聊检测，是闲聊则直接返回固定回复终止
    1. resolve_anaphora      — 指代消解，结合对话历史补全模糊代词
    2. analyze_and_decompose — 复杂问题拆解 + 信息完整性判断
    """

    # 固定闲聊回复
    CHAT_REPLY = "你好～我可以为你解答知识库相关问题，请描述你的业务问题哦"

    def __init__(self) -> None:
        self._chat_llm = _create_intent_llm(temperature=0)
        self._resolve_llm = _create_intent_llm(temperature=0)
        self._analyze_llm = _create_intent_llm(temperature=0)

    # ---- 节点 0: 闲聊检测（最先执行） ----

    async def classify_chat(self, state: IntentState) -> Dict[str, Any]:
        """检测用户输入是否为闲聊，是则短路整个管道。"""
        user_query = state.get("user_query", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        logger.info("[IntentAgent.Chat] detecting: %s", user_query[:80])

        try:
            chain = CHAT_DETECT_PROMPT | self._chat_llm
            response = await chain.ainvoke({"user_query": user_query})
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            is_chat = bool(result.get("is_chat", False))
        except Exception as e:
            logger.warning("[IntentAgent.Chat] LLM failed (%s), default to non-chat.", e)
            is_chat = False

        if is_chat:
            agent_log.append(f"👋 闲聊检测: 是 → 短路终止")
            logger.info("[IntentAgent.Chat] is_chat=True, short-circuit")
        else:
            agent_log.append("💬 闲聊检测: 否 → 继续知识问答流程")
            logger.info("[IntentAgent.Chat] is_chat=False, continue")

        return {
            "is_chat": is_chat,
            "agent_log": agent_log,
        }

    # ---- 节点 0.1: 闲聊回复 ----

    async def chat_response(self, state: IntentState) -> Dict[str, Any]:
        """返回固定闲聊回复，直接终止。"""
        agent_log: list[str] = list(state.get("agent_log", []))
        agent_log.append(f"👋 回复: {self.CHAT_REPLY}")

        return {
            "chat_reply": self.CHAT_REPLY,
            "route_action": "end",
            "agent_log": agent_log,
        }

    # ---- 节点 1: 指代消解 ----

    async def resolve_anaphora(self, state: IntentState) -> Dict[str, Any]:
        """结合对话历史，将模糊代词补全为独立完整问题。"""
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        logger.info("[IntentAgent.Resolve] query=%s", user_query[:80])

        # 无历史 → 直接透传
        if not chat_history:
            agent_log.append("💬 指代消解: 无历史对话，跳过")
            return {
                "resolved_query": user_query,
                "agent_log": agent_log,
            }

        # 有历史 → LLM 消解
        try:
            history_text = "\n".join(
                f"[{h['role']}]: {h['content'][:200]}" for h in chat_history[-6:]
            )
            chain = ANAPHORA_PROMPT | self._resolve_llm
            response = await chain.ainvoke({
                "history": history_text,
                "user_query": user_query,
            })
            resolved = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
        except Exception as e:
            logger.warning("[IntentAgent.Resolve] LLM failed (%s), using original.", e)
            resolved = user_query

        if resolved != user_query:
            agent_log.append(f"💬 指代消解: {user_query[:40]} → {resolved[:60]}")
        else:
            agent_log.append("💬 指代消解: 无需改写")

        logger.info("[IntentAgent.Resolve] resolved=%s", resolved[:80])

        return {
            "resolved_query": resolved,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 拆解 + 澄清判断 ----

    async def analyze_and_decompose(self, state: IntentState) -> Dict[str, Any]:
        """复杂问题拆解为 query_list，同时判断是否需要澄清。"""
        resolved_query = state.get("resolved_query", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        logger.info("[IntentAgent.Analyze] analyzing: %s", resolved_query[:80])

        try:
            chain = DECOMPOSE_AND_CLARIFY_PROMPT | self._analyze_llm
            response = await chain.ainvoke({"resolved_query": resolved_query})
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            # 去除可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            query_list: list = result.get("query_list", [])
            need_clarify: bool = bool(result.get("need_clarify", False))
            clarify_msg: str = result.get("clarify_msg", "")

        except Exception as e:
            logger.warning("[IntentAgent.Analyze] Parse failed (%s), fallback.", e)
            query_list = [{"query": resolved_query}]
            need_clarify = False
            clarify_msg = ""

        # 日志
        if need_clarify:
            agent_log.append(f"⚠️ 需要澄清: {clarify_msg[:80]}")
        else:
            agent_log.append(
                f"📋 拆解完成: {len(query_list)} 个子查询"
            )
            for i, sq in enumerate(query_list):
                agent_log.append(f"   Q{i+1}: {sq['query'][:80]}")

        logger.info(
            "[IntentAgent.Analyze] clarify=%s queries=%d",
            need_clarify, len(query_list),
        )

        # 确定路由
        route_action = "retriever_agent"  # 默认下游

        return {
            "query_list": query_list,
            "need_clarify": need_clarify,
            "clarify_msg": clarify_msg,
            "route_action": route_action,
            "agent_log": agent_log,
        }


# ============================================================================
# 子图构建
# ============================================================================

def build_intent_agent():
    """构建并编译 IntentAgent 子图。

    拓扑：
        START → classify_chat
                   │
            is_chat=True  → chat_response → END (短路)
            is_chat=False → resolve_anaphora → analyze_and_decompose → END
    """
    agent = IntentAgent()

    workflow = StateGraph(IntentState)

    workflow.add_node("classify_chat", agent.classify_chat)
    workflow.add_node("chat_response", agent.chat_response)
    workflow.add_node("resolve_anaphora", agent.resolve_anaphora)
    workflow.add_node("analyze_and_decompose", agent.analyze_and_decompose)

    workflow.set_entry_point("classify_chat")

    # 闲聊分支
    workflow.add_conditional_edges(
        "classify_chat",
        lambda s: "chat_response" if s.get("is_chat") else "resolve_anaphora",
        {"chat_response": "chat_response", "resolve_anaphora": "resolve_anaphora"},
    )
    workflow.add_edge("chat_response", END)

    # 知识问答路径
    workflow.add_edge("resolve_anaphora", "analyze_and_decompose")
    workflow.add_edge("analyze_and_decompose", END)

    compiled = workflow.compile()

    logger.info(
        "IntentAgent subgraph compiled. "
        "Topology: START → classify_chat → "
        "[chat_response → END | resolve_anaphora → analyze_and_decompose → END]"
    )

    return compiled
