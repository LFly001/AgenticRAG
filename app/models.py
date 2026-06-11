"""请求与响应 Pydantic 模型定义。"""
from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """问答请求模型。"""
    question: str
    session_id: str = ""  # 空字符串 = 单轮模式


class SourceInfo(BaseModel):
    """引用来源详细信息。"""
    id: str
    source_file: str
    page: Optional[str] = None
    type: Optional[str] = None
    snippet: Optional[str] = None


class RetrievalDetails(BaseModel):
    """检索过程详细信息。"""
    doc_count: int
    rerank_scores: list[float]


class QueryResponse(BaseModel):
    """Agentic RAG 问答响应模型。"""
    answer: str
    sources: list[SourceInfo]
    retrieval_details: RetrievalDetails
    thought_process: list[str] = []
    session_id: str = ""


class DocumentInfo(BaseModel):
    """知识库文档摘要信息。"""
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """文档列表响应。"""
    documents: list[DocumentInfo]
    total_chunks: int


class DeleteResponse(BaseModel):
    """删除操作响应。"""
    success: bool
    message: str
    deleted_chunks: int = 0
