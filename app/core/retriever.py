"""混合检索器 — 向量 + BM25 + RRF 融合 + CrossEncoder 重排序。

入库逻辑已提取至 app.core.indexer.IndexingMixin。
"""

import asyncio
from typing import List, Dict, Any, Optional

import bm25s
import chromadb
import jieba
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import CrossEncoder

from app.config import settings
from app.core.embedding import HybridEmbeddings
from app.core.indexer import IndexingMixin
from app.utils.logger import get_logger

# 抑制 jieba 首次加载时向 stdout 输出 "Building prefix dict..." 等内部日志
jieba.setLogLevel(jieba.logging.WARNING)

logger = get_logger(__name__)


class HybridRetriever(IndexingMixin):
    """混合检索器 — 向量 + BM25 + RRF 融合 + CrossEncoder 重排序。

    继承 IndexingMixin 获得入库能力（add_documents_to_index 等）。
    """

    def __init__(self):
        logger.info(
            "Initializing ChromaDB at %s", settings.CHROMA_PERSIST_DIR
        )
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection_name = settings.COLLECTION_NAME
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        try:
            logger.info(
                "Initializing Embedding Model via HybridEmbeddings wrapper..."
            )
            self.embedder = HybridEmbeddings()
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise e

        logger.info(
            "Loading Re-ranker from local path: %s",
            settings.RERANKER_MODEL_NAME,
        )
        try:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
            logger.info("Re-ranker loaded successfully.")
        except Exception as e:
            logger.warning(
                "Re-ranker load failed: %s. Will skip reranking.", e
            )
            self.reranker = None

        # --- BM25S 初始化 ---
        self.bm25_retriever: Optional[bm25s.BM25] = None
        self.bm25_corpus_ids: List[str] = []
        self.bm25_corpus_texts: List[str] = []

        # 使用锁保护 BM25 索引的更新操作
        self.bm25_lock = asyncio.Lock()

        # 初始化 BM25S 索引（来自 IndexingMixin）
        self._init_bm25s_from_db()

    # ========================================================================
    # 分词
    # ========================================================================

    def _tokenize(self, text: str) -> List[str]:
        """中文用 jieba 分词，否则空格分词。"""
        if any('一' <= char <= '鿿' for char in text):
            return list(jieba.cut(text))
        else:
            return text.split()

    # ========================================================================
    # 检索入口
    # ========================================================================

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """混合检索 — 向量 + BM25 + RRF 融合 + CrossEncoder 重排序。

        Args:
            query: 查询文本
            top_k: 最终返回文档数（默认使用配置值）
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL

        return await self._retrieve_hybrid(query, top_k)

    # ========================================================================
    # 混合检索（向量 + BM25 + RRF + Reranker）
    # ========================================================================

    async def _retrieve_hybrid(
        self, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """向量 + BM25 + RRF 融合 + CrossEncoder 重排序。"""
        current_bm25 = self.bm25_retriever
        current_ids = self.bm25_corpus_ids

        if current_bm25 is None or not current_ids:
            return []

        loop = asyncio.get_running_loop()

        # Step 1: Vector Retrieval
        query_embedding = await self.embedder.aembed_query(query)
        count = self.collection.count()
        n_results = min(top_k * 2, count)
        if count == 0:
            return []

        vector_results = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            ),
        )

        vec_ids = vector_results['ids'][0]
        vec_rank_map = {
            id_: rank + 1 for rank, id_ in enumerate(vec_ids)
        }

        # 直接从 query 结果构建 lookup，避免对向量命中 ID 的二次 ChromaDB 查询
        vec_data: Dict[str, Dict[str, Any]] = {
            id_: {"text": text, "metadata": meta}
            for id_, text, meta in zip(
                vector_results['ids'][0],
                vector_results['documents'][0],
                vector_results['metadatas'][0],
            )
        }

        # Step 2: BM25S Retrieval
        query_tokens = self._tokenize(query)
        bm25_ids: List[str] = []
        try:
            indices, scores = current_bm25.retrieve(
                [query_tokens], k=n_results
            )
            if indices.size > 0:
                valid_indices = indices[0][scores[0] > 0]
                bm25_ids = [
                    current_ids[i]
                    for i in valid_indices
                    if i < len(current_ids)
                ]
        except Exception as e:
            logger.error("BM25S retrieval error: %s", e)

        bm25_rank_map = {
            id_: rank + 1 for rank, id_ in enumerate(bm25_ids)
        }

        # Step 3: RRF (Reciprocal Rank Fusion)
        k = settings.RRF_K_CONSTANT
        all_candidate_ids = set(vec_ids) | set(bm25_ids)
        rrf_scores = {}
        for doc_id in all_candidate_ids:
            vec_rank = vec_rank_map.get(doc_id, len(vec_ids) + 1)
            bm25_rank = bm25_rank_map.get(doc_id, len(bm25_ids) + 1)
            score = (1.0 / (k + vec_rank)) + (1.0 / (k + bm25_rank))
            rrf_scores[doc_id] = score

        sorted_rrf_items = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        candidate_ids = [item[0] for item in sorted_rrf_items]

        if not candidate_ids:
            return []

        # 只对 BM25 独有（向量结果中不存在）的 ID 发起 ChromaDB 补充查询
        bm25_only_ids = [cid for cid in candidate_ids if cid not in vec_data]
        bm25_data: Dict[str, Dict[str, Any]] = {}
        if bm25_only_ids:
            bm25_retrieved = await loop.run_in_executor(
                None,
                lambda: self.collection.get(
                    ids=bm25_only_ids, include=["documents", "metadatas"]
                ),
            )
            for id_, text, meta in zip(
                bm25_retrieved['ids'],
                bm25_retrieved['documents'],
                bm25_retrieved['metadatas'],
            ):
                bm25_data[id_] = {"text": text, "metadata": meta}

        # 合并向量 lookup + BM25 补充数据构建候选文档
        candidate_docs = []
        for i, doc_id in enumerate(candidate_ids):
            data = vec_data.get(doc_id) or bm25_data.get(doc_id)
            if data is None:
                continue
            candidate_docs.append({
                "id": doc_id,
                "text": data["text"],
                "metadata": data["metadata"],
                "rrf_score": sorted_rrf_items[i][1],
                "strategy": "hybrid",
            })

        # Step 4: Re-ranking
        final_results = candidate_docs
        if self.reranker and candidate_docs:
            try:
                pairs = [[query, doc["text"]] for doc in candidate_docs]
                rerank_scores = await loop.run_in_executor(
                    None, lambda: self.reranker.predict(pairs)
                )
                for i, doc in enumerate(candidate_docs):
                    doc["rerank_score"] = float(rerank_scores[i])
                final_results = sorted(
                    candidate_docs,
                    key=lambda x: x["rerank_score"],
                    reverse=True,
                )
            except Exception as e:
                logger.error("Re-ranking failed: %s", e)

        final_top_k = settings.TOP_K_RERANK
        return final_results[:final_top_k]
