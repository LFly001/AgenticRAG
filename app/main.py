"""多 Agent 协作智能知识库 — FastAPI 应用主入口。

路由：
  POST /upload          文档上传 + 后台解析入库
  GET  /task-status/{id} 后台任务状态查询
  POST /query           统一问答入口（5-Agent 协作图）

Agent 协作流程：
  ConversationAgent（意图分类+对话记忆）→ QueryPlanner → RetrieverAgent(×N)
        → ResponderAgent → CriticAgent → [regenerate]
"""

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import (
    BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile,
)

from app.core.chunker import SmartChunker
from app.core.parser import DocumentParser
from app.core.retriever import HybridRetriever
from app.graph import build_graph
from app.models import QueryRequest, QueryResponse, RetrievalDetails, SourceInfo
from app.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = "./data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 后台任务状态字典，受 _task_status_lock 保护
task_status: dict[str, str] = {}
_task_status_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：初始化基础组件 + 构建多 Agent 协作图。"""
    logger.info("Initializing application resources...")

    # 1. 文档处理管道
    app.state.parser = DocumentParser()
    app.state.chunker = SmartChunker()

    # 2. 混合检索器（向量 + BM25 + RRF 融合 + CrossEncoder 重排序）
    app.state.retriever = HybridRetriever()

    # 3. 构建 5-Agent 协作图（5 子图 + 8 父图节点）
    logger.info("Building 5-Agent collaboration graph...")
    app.state.graph = build_graph(app.state.retriever)
    logger.info(
        "5-Agent graph ready (ConversationAgent + QueryPlanner + "
        "RetrieverAgent + ResponderAgent + CriticAgent)."
    )

    logger.info("All resources initialized successfully.")
    yield
    logger.info("Shutting down application...")


app = FastAPI(title="Multi-Agent Intelligent Knowledge Base", lifespan=lifespan)


async def _set_task_status(task_id: str, status: str) -> None:
    """协程安全的写入任务状态。"""
    async with _task_status_lock:
        task_status[task_id] = status


async def _get_task_status(task_id: str) -> str:
    """协程安全的读取任务状态。"""
    async with _task_status_lock:
        return task_status.get(task_id, "not_found")


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    status = await _get_task_status(task_id)
    return {"task_id": task_id, "status": status}


async def process_document_background(file_path: str, filename: str, task_id: str):
    """
    后台任务：解析 → 切片 → 入库
    支持覆盖更新：如果文件名已存在，先删除旧数据。
    """
    loop = asyncio.get_running_loop()

    try:
        await _set_task_status(task_id, "processing")
        logger.info(f"[Task {task_id}] Starting background processing for {filename}")

        # 1. 检查并删除旧数据（实现覆盖效果）
        logger.info(f"Checking for existing documents with name: {filename}")
        await app.state.retriever.delete_documents_by_source(filename)

        # 2. 解析文档
        parsed_docs = await loop.run_in_executor(None, app.state.parser.parse_file, file_path)

        if not parsed_docs:
            raise ValueError("Failed to parse document or empty content")

        # 3. 智能切片
        chunks = await loop.run_in_executor(None, app.state.chunker.chunk_documents, parsed_docs)

        if not chunks:
            raise ValueError("No valid chunks generated from document")

        # 4. 入库
        await app.state.retriever.add_documents_to_index(chunks)

        await _set_task_status(task_id, "completed")
        logger.info(f"[Task {task_id}] Successfully indexed {len(chunks)} chunks.")

    except Exception as e:
        logger.error(f"[Task {task_id}] Processing failed: {e}", exc_info=True)
        await _set_task_status(task_id, f"failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                await loop.run_in_executor(None, os.remove, file_path)
                logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if ".." in file.filename or file.filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    task_id = str(uuid.uuid4())

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: shutil.copyfileobj(file.file, open(file_path, "wb")),
        )

        logger.info(f"File saved to {file_path}, Task ID: {task_id}")

        background_tasks.add_task(process_document_background, file_path, file.filename, task_id)

        logger.info(f"[Task {task_id}] Returning upload success response to client.")
        return {
            "message": "File uploaded successfully. Processing started in background.",
            "task_id": task_id,
            "filename": file.filename,
        }

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        if os.path.exists(file_path):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, os.remove, file_path)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge(query_request: QueryRequest, req: Request):
    """统一问答入口 — 5-Agent 协作图（含对话记忆）。

    完整执行路径：
    1. ConversationAgent 意图分类 + 对话记忆 + 追问融合（含闲聊直接回复）
    2. QueryPlanner 复杂度分析 + 复合问题拆解
    3. RetrieverAgent 策略选择 → 检索 → 自评 → 改写
    4. ResponderAgent 上下文构建 → LLM 生成 → 引用解析
    5. CriticAgent 三维度评审 → pass / fail → regenerate
    6. save_conversation 保存问答到 Redis 记忆
    """
    try:
        graph = req.app.state.graph

        logger.info(f"[MultiAgent] Processing: {query_request.question[:80]}...")

        # 执行 LangGraph 工作流
        result = await graph.ainvoke({
            "question": query_request.question,
            "session_id": query_request.session_id,
        })

        final_answer = result.get("final_answer", "")
        sources = result.get("sources", [])
        retrieval_details = result.get("retrieval_details", {})
        node_log = result.get("node_log", [])

        # 构建 SourceInfo 列表
        source_objects = [
            SourceInfo(
                id=src.get("id", ""),
                source_file=src.get("source_file", "Unknown"),
                page=str(src.get("page", "N/A")),
                type=src.get("type", "Text"),
                snippet=src.get("snippet", ""),
            )
            for src in sources
        ]

        retrieval_details_obj = RetrievalDetails(
            doc_count=retrieval_details.get("doc_count", len(source_objects)),
            rerank_scores=retrieval_details.get("rerank_scores", []),
        )

        logger.info(
            f"[MultiAgent] Complete. Path: {' → '.join(node_log)}"
        )

        return QueryResponse(
            answer=final_answer,
            sources=source_objects,
            retrieval_details=retrieval_details_obj,
            thought_process=node_log,
            session_id=query_request.session_id,
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, loop="uvloop")
