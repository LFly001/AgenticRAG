"""8-Agent 协作模块。

Agent 清单：
┌──────────────────────┬──────────────────────────────────┐
│ Agent                │ 职责                              │
├──────────────────────┼──────────────────────────────────┤
│ OrchestratorAgent    │ 总调度，唯一入口，全局分支分发      │
│ IntentAgent          │ 意图解析，澄清判断                 │
│ RetrieveAgent        │ 检索调度，多策略召回               │
│ DocFilterAgent       │ 文档校验清洗，去重过滤             │
│ ContextCompressAgent │ 上下文压缩，token 预算管理         │
│ ReasonAgent          │ 逻辑推理，思维链生成               │
│ WriterAgent          │ 答案生成，引用规范                 │
│ AntiHallucinationAgent│ 幻觉检测，事实校验                │
└──────────────────────┴──────────────────────────────────┘

图拓扑：

    START
      │
      ▼
   orchestrator ──→ route_dispatcher（全局分支分发）
      │                  ▲
      ▼                  │
   intent_agent ─────────┤
      │                  │
      ▼                  │
   retriever_agent ──────┤
      │                  │
      ▼                  │
   doc_filter_agent ─────┤
      │                  │
      ▼                  │
   context_compress ─────┤
      │                  │
      ▼                  │
   reason_agent ─────────┤
      │                  │
      ▼                  │
   writer_agent ─────────┤
      │                  │
      ▼                  │
   anti_hallucination ───┘
      │
      ▼
     END
"""

from app.graph.agents.orchestrator import (
    OrchestratorAgent,
    build_orchestrator_agent,
    route_dispatcher,
    OrchestratorState,
)
from app.graph.agents.intent import (
    IntentAgent,
    build_intent_agent,
    IntentState,
)
from app.graph.agents.retriever import (
    RetrieveAgent,
    build_retriever_agent,
    RetrieverState,
)
from app.graph.agents.doc_filter import (
    DocFilterAgent,
    build_doc_filter_agent,
    DocFilterState,
)
from app.graph.agents.context_compress import (
    ContextCompressAgent,
    build_context_compress_agent,
    ContextCompressState,
)
from app.graph.agents.reason import (
    ReasonAgent,
    build_reason_agent,
    ReasonState,
)
from app.graph.agents.writer import (
    WriterAgent,
    build_writer_agent,
    WriterState,
)
from app.graph.agents.anti_hallucination import (
    AntiHallucinationAgent,
    build_anti_hallucination_agent,
    AntiHallucinationState,
)

__all__ = [
    # Orchestrator
    "OrchestratorAgent", "build_orchestrator_agent", "route_dispatcher", "OrchestratorState",
    # Intent
    "IntentAgent", "build_intent_agent", "IntentState",
    # Retriever
    "RetrieveAgent", "build_retriever_agent", "RetrieverState",
    # DocFilter
    "DocFilterAgent", "build_doc_filter_agent", "DocFilterState",
    # ContextCompress
    "ContextCompressAgent", "build_context_compress_agent", "ContextCompressState",
    # Reason
    "ReasonAgent", "build_reason_agent", "ReasonState",
    # Writer
    "WriterAgent", "build_writer_agent", "WriterState",
    # AntiHallucination
    "AntiHallucinationAgent", "build_anti_hallucination_agent", "AntiHallucinationState",
]
