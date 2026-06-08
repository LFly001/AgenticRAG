"""ConversationAgent — 用户意图理解 + 对话记忆 + 上下文融合。

内部工作流（6 节点）：

    START
      │
      ▼
   classify_intent ─→ LLM 一次判断: direct_answer | knowledge_query
      │
      ├── direct_answer ─→ 直接回答 ──→ END
      │
      └── knowledge_query → load_history → classify_followup
                                │
                                ├── 非追问 → skip_enrich ──→ END
                                └── 是追问 → enrich_question ──→ END
"""

from __future__ import annotations

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.stores.session_store import conv_store
from app.graph.agents.conversation.state import ConversationState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# LLM
# ============================================================================


def _create_conversation_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
        max_tokens=512,
        request_timeout=20,
        max_retries=1,
    )


# ============================================================================
# Prompts
# ============================================================================

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """你是一个用户意图分析专家。判断用户问题属于哪一类。

- **direct_answer**：简单闲聊、问候、感谢、告别，或不需要查询任何知识库的常识性问题。
  例如："你好"、"谢谢"、"今天天气怎么样"、"再见"
- **knowledge_query**：需要查询企业知识库才能回答的问题。
  例如："公司年假政策是什么"、"2024 Q3 营收多少"、"A产品和B产品对比"

请**只返回一个 JSON 对象**：
{{"intent": "<direct_answer 或 knowledge_query>", "reason": "<一句话>"}}

Question: {question}

JSON:"""
)

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """你是一个企业级智能助手。请直接、友好地回答用户。

对话历史:
{history}

Question: {question}

Answer:"""
)

FOLLOWUP_PROMPT = ChatPromptTemplate.from_template(
    """你是一个对话分析专家。判断用户当前问题是否为对上一轮对话的追问。

对话历史:
{history}

当前问题: {question}

判断标准：
- 追问（followup）：使用了代词指代上文（如"它""那个""第二个"），或省略了上文已提及的主语/宾语，或是对上一轮回答的进一步追问（"详细说说""具体呢""为什么"）
- 新问题（new）：引入了全新话题，不依赖上文即可独立理解

请**只返回一个 JSON 对象**：
{{"is_followup": true或false, "reason": "一句话"}}

JSON:"""
)

ENRICH_PROMPT = ChatPromptTemplate.from_template(
    """你是一个对话上下文融合专家。用户提出了一个追问，需要结合历史对话来还原其完整意图。

对话历史:
{history}

用户追问: {question}

请将用户追问改写为一个独立、完整的问题，使其不依赖历史上下文即可被理解。

规则：
1. 将代词（"它""那个""这个""他"等）替换为具体的实体名称
2. 补全被省略的主语、宾语、定语
3. 如果追问含糊（"详细说说"），结合历史推断具体要详细说明什么
4. 保持原始问题的语言和语气

请**只返回改写后的问题文本**，不要包含任何解释。

改写后的问题:"""
)


# ============================================================================
# ConversationAgent
# ============================================================================


class ConversationAgent:
    """ConversationAgent — 用户意图理解 + 对话记忆 + 上下文融合。

    替换了原先分离的 router + ConversationAgent 设计，
    一次 LLM 调用完成闲聊/知识问答二分。
    """

    def __init__(self) -> None:
        self._intent_llm = _create_conversation_llm(temperature=0)
        self._answer_llm = _create_conversation_llm(temperature=0.3)
        self._followup_llm = _create_conversation_llm(temperature=0)
        self._enrich_llm = _create_conversation_llm(temperature=0)

    # ---- 节点 1: 意图分类 ----

    async def classify_intent(self, state: ConversationState) -> Dict[str, Any]:
        """LLM 判断用户意图：闲聊 or 知识问答。"""
        question = state["question"]
        agent_log: list[str] = []

        try:
            chain = INTENT_PROMPT | self._intent_llm
            response = await chain.ainvoke({"question": question})
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            intent = result.get("intent", "knowledge_query")
            reason = result.get("reason", "N/A")
        except Exception as e:
            logger.warning("[Conversation] classify_intent failed (%s), defaulting.", e)
            intent = "knowledge_query"
            reason = f"error: {e}"

        agent_log.append(f"💬 意图: {intent}（{reason}）")
        logger.info("[Conversation] intent=%s", intent)

        return {
            "intent": intent,
            "agent_log": agent_log,
        }

    # ---- 节点 2: 直接回答 ----

    async def direct_answer(self, state: ConversationState) -> Dict[str, Any]:
        """闲聊路径：直接生成回答。"""
        question = state["question"]
        session_id = state.get("session_id", "")
        agent_log = list(state.get("agent_log", []))

        history = conv_store.get_history(session_id) if session_id else []
        history_text = "\n".join(
            f"[{h['role']}]: {h['content'][:100]}" for h in history[-6:]
        ) or "(无历史)"

        chain = DIRECT_ANSWER_PROMPT | self._answer_llm
        response = await chain.ainvoke({
            "history": history_text,
            "question": question,
        })
        answer = response.content if hasattr(response, "content") else str(response)

        agent_log.append("💬 直接回答（无需检索）")
        return {
            "answer": answer,
            "sources": [],
            "retrieval_details": {"doc_count": 0},
            "agent_log": agent_log,
        }

    # ---- 节点 3: 加载历史 ----

    async def load_history(self, state: ConversationState) -> Dict[str, Any]:
        """从 Redis 加载对话历史。"""
        session_id = state.get("session_id", "")
        question = state["question"]
        agent_log = list(state.get("agent_log", []))

        if not session_id:
            agent_log.append("💬 无会话 ID，单轮模式")
            return {
                "chat_history": [],
                "is_followup": False,
                "enriched_question": question,
                "agent_log": agent_log,
            }

        history = conv_store.get_history(session_id)
        agent_log.append(f"💬 加载历史: {len(history) // 2} 轮对话")

        if not history:
            return {
                "chat_history": [],
                "is_followup": False,
                "enriched_question": question,
                "agent_log": agent_log,
            }

        return {"chat_history": history, "agent_log": agent_log}

    # ---- 节点 4: 判断追问 ----

    async def classify_followup(self, state: ConversationState) -> Dict[str, Any]:
        """LLM 判断当前问题是否为追问。"""
        question = state["question"]
        history = state.get("chat_history", [])
        agent_log = list(state.get("agent_log", []))

        if not history:
            return {
                "is_followup": False,
                "enriched_question": question,
                "agent_log": agent_log,
            }

        try:
            history_text = "\n".join(
                f"[{h['role']}]: {h['content'][:200]}" for h in history[-6:]
            )
            chain = FOLLOWUP_PROMPT | self._followup_llm
            response = await chain.ainvoke({
                "history": history_text, "question": question,
            })
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            is_followup = bool(result.get("is_followup", False))
            reason = result.get("reason", "N/A")
        except Exception as e:
            logger.warning("[Conversation] classify_followup failed (%s)", e)
            is_followup = False
            reason = f"error: {e}"

        agent_log.append(f"💬 追问: {'是' if is_followup else '否'}（{reason}）")
        return {"is_followup": is_followup, "agent_log": agent_log}

    # ---- 节点 5: 融合上下文 ----

    async def enrich_question(self, state: ConversationState) -> Dict[str, Any]:
        """将追问改写为独立完整的问题。"""
        question = state["question"]
        history = state.get("chat_history", [])
        agent_log = list(state.get("agent_log", []))

        try:
            history_text = "\n".join(
                f"[{h['role']}]: {h['content'][:200]}" for h in history[-6:]
            )
            chain = ENRICH_PROMPT | self._enrich_llm
            response = await chain.ainvoke({
                "history": history_text, "question": question,
            })
            enriched = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
        except Exception as e:
            logger.warning("[Conversation] enrich failed (%s)", e)
            enriched = question

        agent_log.append(f"💬 融合: {question[:40]} → {enriched[:60]}")
        return {"enriched_question": enriched, "agent_log": agent_log}

    # ---- 节点 6: 跳过融合 ----

    async def skip_enrich(self, state: ConversationState) -> Dict[str, Any]:
        agent_log = list(state.get("agent_log", []))
        agent_log.append("💬 非追问，使用原始问题")
        return {"enriched_question": state["question"], "agent_log": agent_log}


# ============================================================================
# 子图构建
# ============================================================================


def build_conversation_agent():
    agent = ConversationAgent()

    workflow = StateGraph(ConversationState)

    workflow.add_node("classify_intent", agent.classify_intent)
    workflow.add_node("direct_answer", agent.direct_answer)
    workflow.add_node("load_history", agent.load_history)
    workflow.add_node("classify_followup", agent.classify_followup)
    workflow.add_node("enrich_question", agent.enrich_question)
    workflow.add_node("skip_enrich", agent.skip_enrich)

    workflow.set_entry_point("classify_intent")

    # 意图分支：闲聊 or 知识问答
    workflow.add_conditional_edges(
        "classify_intent", _route_after_intent,
        {"direct_answer": "direct_answer", "load_history": "load_history"},
    )
    workflow.add_edge("direct_answer", END)

    # 知识问答路径
    workflow.add_edge("load_history", "classify_followup")
    workflow.add_conditional_edges(
        "classify_followup", _route_after_followup,
        {"enrich_question": "enrich_question", "skip_enrich": "skip_enrich"},
    )
    workflow.add_edge("enrich_question", END)
    workflow.add_edge("skip_enrich", END)

    compiled = workflow.compile()

    logger.info(
        "ConversationAgent subgraph compiled (6 nodes). "
        "Topology: START → classify → [direct_answer → END | "
        "load → classify → [enrich|skip] → END]"
    )
    return compiled


def _route_after_intent(state: ConversationState) -> str:
    if state.get("intent") == "direct_answer":
        return "direct_answer"
    return "load_history"


def _route_after_followup(state: ConversationState) -> str:
    if state.get("is_followup"):
        return "enrich_question"
    return "skip_enrich"
