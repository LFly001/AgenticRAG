"""IntentAgent — 意图解析节点。

职责：
0. 闲聊检测：最先执行，识别问候/感谢/告别等非业务对话 → 短路终止
1. 查询理解：一次 LLM 调用完成指代消解 + 复杂度判断 + 子查询拆解 + 澄清判断

内部工作流（3 节点）：

    START
      │
      ▼
   classify_chat ─→ 闲聊检测（LLM #1，短路入口）
      │
      ├── is_chat=True ──→ chat_response ──→ END（返回固定回复，跳过后续全链路）
      │
      └── is_chat=False → understand_query ──→ END  （LLM #2，一步到位）

路由分支（由父图 route_dispatcher 处理）：
    - is_chat=True       → 结束流程，返回固定闲聊回复
    - need_clarify=True  → 结束流程，返回澄清话术给用户
    - need_clarify=False → 下游 retriever_agent
"""

from __future__ import annotations

from typing import Dict, Any

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.core.llm_factory import get_llm
from app.graph.agents.intent.state import IntentState
from app.utils.llm_utils import parse_llm_json
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
- 即使带有礼貌用语（如"请问一下宪法第一条是什么"），只要包含实质性查询内容，就不是闲聊

用户输入: {user_query}

请**只返回一个 JSON 对象**：
{{"is_chat": true或false}}

JSON:""")


QUERY_UNDERSTAND_PROMPT = ChatPromptTemplate.from_template("""你是一个法律查询理解专家。请同时完成以下任务：指代消解、复杂度判断、子查询拆解、信息完整性检查。

<对话历史>
{history}
</对话历史>

<用户当前输入>
{user_query}
</用户当前输入>

## 任务 1：指代消解
结合对话历史，将用户问题还原为独立完整的问题：
1. 将代词（"它""那个""这个""他""她""第二条""上一款""前一条"等）替换为历史中对应的具体法律概念或实体名称
2. 补全被省略的主语、宾语、定语
3. 如果用户追问含糊（"详细说说""具体呢""为什么"），结合历史推断具体要说明什么
4. 如果用户输入已是一个独立完整的问题，原样返回
5. 如果无对话历史或历史不相关，直接将 user_query 作为 resolved_query

## 任务 2：复杂度判断 + 子查询拆解

**简单问题**（query_list 只包含一项，即 resolved_query 本身）：
- 单一法条查询："宪法第一条的内容是什么？"
- 单一概念解释："什么是表见代理？"
- 单一流程："民事诉讼的上诉流程是怎样的？"
- 单一事实追问："盗窃罪的量刑标准是什么？"

**复杂问题**（需要拆解为多个子查询，每项必须能独立检索）：
- 多法条比较："对比刑法第232条故意杀人和第233条过失致人死亡的构成要件"
- 多情形分析："故意伤害和过失伤害的构成要件分别是什么？"
- 条件分支："如果合同无效，双方责任如何划分？如果合同可撤销呢？"
- 多跳推理："民法典中关于高空抛物的规定是什么？侵权责任如何认定？"
- 多维度："商标侵权的认定标准、赔偿计算方式和诉讼时效各是什么？"

拆解规则：
- 每个子查询必须能独立检索
- 子查询之间应有清晰的逻辑边界
- 子查询数量不超过 5 个

## 任务 3：信息完整性检查
检查用户问题是否存在：关键实体缺失、条件不明确、指代无法消解、问题过于模糊。
如果存在 → need_clarify=true，clarify_msg 写一句友好的追问；否则 need_clarify=false，clarify_msg 留空。

## 输出格式

请**只返回一个 JSON 对象**：
{{"resolved_query": "指代消解后的完整问题", "query_list": [{{"query": "子查询1"}}, {{"query": "子查询2"}}], "need_clarify": false, "clarify_msg": ""}}

JSON:""")


# ============================================================================
# IntentAgent 类
# ============================================================================

class IntentAgent:
    """IntentAgent — 意图解析专家。

    内部流水线（闲聊优先短路，查询理解一步到位）：
    0. classify_chat    — 闲聊检测，是闲聊则直接返回固定回复终止
    1. understand_query — 指代消解 + 复杂度判断 + 子查询拆解 + 澄清检查（一次 LLM）
    """

    # 固定闲聊回复
    CHAT_REPLY = "你好～我可以为你解答知识库相关问题，请描述你的业务问题哦"

    def __init__(self) -> None:
        # 共享同一 LLM 实例（temperature=0，解析必须稳定一致）
        self._llm = get_llm(temperature=0, max_tokens=1024, timeout=30)

    # ---- 节点 0: 闲聊检测（最先执行） ----

    async def classify_chat(self, state: IntentState) -> Dict[str, Any]:
        """检测用户输入是否为闲聊，是则短路整个管道。"""
        user_query = state.get("user_query", "")
        agent_log: list[str] = list(state.get("agent_log", []))

        logger.info("[IntentAgent.Chat] detecting: %s", user_query[:80])

        try:
            chain = CHAT_DETECT_PROMPT | self._llm
            response = await chain.ainvoke({"user_query": user_query})
            raw = response.content if hasattr(response, "content") else str(response)
            result = parse_llm_json(raw)
            is_chat = bool(result.get("is_chat", False))
        except Exception as e:
            logger.warning("[IntentAgent.Chat] LLM failed (%s), default to non-chat.", e)
            is_chat = False

        if is_chat:
            agent_log.append("👋 闲聊检测: 是 → 短路终止")
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

    # ---- 节点 1: 查询理解（一次 LLM 完成指代消解 + 拆解 + 澄清检查） ----

    async def understand_query(self, state: IntentState) -> Dict[str, Any]:
        """一次 LLM 调用完成：指代消解 + 复杂度判断 + 子查询拆解 + 信息完整性检查。

        无论有无对话历史，统一走 LLM——无历史时只做拆解+澄清，
        有历史时额外做指代消解。
        """
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])
        agent_log: list[str] = list(state.get("agent_log", []))

        logger.info("[IntentAgent.Understand] query=%s history_rounds=%d",
                    user_query[:80], len(chat_history))

        # 构建历史文本（最近 6 轮，单条最多 500 字）
        if chat_history:
            max_msg = 500
            history_text = "\n".join(
                f"[{h['role']}]: {h['content'] if len(h['content']) <= max_msg else h['content'][:max_msg] + '...'}"
                for h in chat_history[-6:]
            )
        else:
            history_text = "（无历史对话）"

        try:
            chain = QUERY_UNDERSTAND_PROMPT | self._llm
            response = await chain.ainvoke({
                "history": history_text,
                "user_query": user_query,
            })
            raw = response.content if hasattr(response, "content") else str(response)
            result = parse_llm_json(raw)

            resolved_query = result.get("resolved_query", user_query).strip()
            query_list: list = result.get("query_list", [{"query": resolved_query}])
            need_clarify: bool = bool(result.get("need_clarify", False))
            clarify_msg: str = result.get("clarify_msg", "")

        except Exception as e:
            logger.warning("[IntentAgent.Understand] LLM failed (%s), fallback.", e)
            resolved_query = user_query
            query_list = [{"query": user_query}]
            need_clarify = False
            clarify_msg = ""

        # 日志
        if resolved_query != user_query:
            agent_log.append(f"💬 指代消解: {user_query[:40]} → {resolved_query[:60]}")
        else:
            agent_log.append("💬 指代消解: 无需改写")

        if need_clarify:
            agent_log.append(f"⚠️ 需要澄清: {clarify_msg[:80]}")
        else:
            nq = len(query_list)
            agent_log.append(f"📋 拆解完成: {nq} 个子查询")
            for i, sq in enumerate(query_list):
                agent_log.append(f"   Q{i+1}: {sq['query'][:80]}")

        logger.info(
            "[IntentAgent.Understand] resolved=%s queries=%d clarify=%s",
            resolved_query[:80], len(query_list), need_clarify,
        )

        route_action = "retriever_agent"

        return {
            "resolved_query": resolved_query,
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

    拓扑（3 节点）：
        START → classify_chat
                   │
            is_chat=True  → chat_response → END (短路)
            is_chat=False → understand_query → END
    """
    agent = IntentAgent()

    workflow = StateGraph(IntentState)

    workflow.add_node("classify_chat", agent.classify_chat)
    workflow.add_node("chat_response", agent.chat_response)
    workflow.add_node("understand_query", agent.understand_query)

    workflow.set_entry_point("classify_chat")

    # 闲聊分支
    workflow.add_conditional_edges(
        "classify_chat",
        lambda s: "chat_response" if s.get("is_chat") else "understand_query",
        {"chat_response": "chat_response", "understand_query": "understand_query"},
    )
    workflow.add_edge("chat_response", END)

    # 查询理解 → 结束
    workflow.add_edge("understand_query", END)

    compiled = workflow.compile()
    return compiled
