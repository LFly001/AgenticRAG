"""LangGraph Agentic RAG 模块。

使用方式：
    from app.graph import build_graph

    graph = build_graph(retriever, generator)
    result = await graph.ainvoke({"question": "用户问题"})
"""

from app.graph.graph import build_graph
from app.graph.state import GraphState

__all__ = ["build_graph", "GraphState"]
