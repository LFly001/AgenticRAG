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
from app.stores.document_store import redis_store
from app.utils.logger import get_logger

# 抑制 jieba 首次加载时向 stdout 输出 "Building prefix dict..." 等内部日志
jieba.setLogLevel(jieba.logging.WARNING)

logger = get_logger(__name__)


class HybridRetriever(IndexingMixin):
    """混合检索器 — 向量 + BM25 + RRF 融合 + CrossEncoder 重排序。

    继承 IndexingMixin 获得入库能力（add_documents_to_index 等）。

    模型（BGE-M3, BGE-Reranker）采用懒加载：首次检索/入库时才加载，
    避免启动阶段阻塞 2+ 分钟。
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

        # 模型懒加载：启动时仅占位，首次使用时通过 _ensure_models() 加载
        self.embedder = None
        self.reranker = None
        self._models_loaded = False
        self._model_lock = asyncio.Lock()

        # --- BM25S 初始化 ---
        self.bm25_retriever: Optional[bm25s.BM25] = None
        self.bm25_corpus_ids: List[str] = []
        self.bm25_corpus_texts: List[str] = []

        # 使用锁保护 BM25 索引的更新操作
        self.bm25_lock = asyncio.Lock()

        # 文档摘要缓存: source_file → chunk_count（增删时维护，避免 ChromaDB 全量查询）
        self._doc_summary: Dict[str, int] = {}

        # 初始化 BM25S 索引（来自 IndexingMixin），并构建文档摘要
        self._init_bm25s_from_db()
        self._build_doc_summary_from_db()

    # ========================================================================
    # 模型懒加载
    # ========================================================================

    async def _ensure_models(self):
        """首次检索/入库时加载嵌入模型和重排序模型（线程池执行，不阻塞事件循环）。

        使用 double-check locking 防止并发请求重复加载。
        """
        if self._models_loaded:
            return

        async with self._model_lock:
            if self._models_loaded:  # double-check
                return

            loop = asyncio.get_running_loop()

            logger.info("Lazy-loading embedding model (BGE-M3)...")
            try:
                self.embedder = await loop.run_in_executor(None, HybridEmbeddings)
            except Exception as e:
                logger.error("Failed to load embedding model: %s", e)
                raise e

            logger.info("Lazy-loading reranker model (BGE-Reranker-v2-M3)...")
            try:
                self.reranker = await loop.run_in_executor(
                    None, lambda: CrossEncoder(settings.RERANKER_MODEL_NAME)
                )
                logger.info("All models loaded successfully.")
            except Exception as e:
                logger.warning(
                    "Re-ranker load failed: %s. Will skip reranking.", e
                )
                self.reranker = None

            self._models_loaded = True

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
        await self._ensure_models()

        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL

        return await self._retrieve_hybrid(query, top_k)

    # ========================================================================
    # 混合检索（向量 + BM25 + RRF + Reranker）
    # ========================================================================

    async def _retrieve_hybrid(
        self, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """向量 + BM25 + RRF 融合 + CrossEncoder 重排序。

        向量检索和 BM25 检索无依赖关系，asyncio.gather 并行执行，
        将检索延迟从 T_vec + T_bm25 降至 max(T_vec, T_bm25)。
        """
        current_bm25 = self.bm25_retriever
        current_ids = self.bm25_corpus_ids

        if current_bm25 is None or not current_ids:
            return []

        loop = asyncio.get_running_loop()

        # Step 0: 获取 query embedding 和 count（向量检索的前置条件）
        query_embedding = await self.embedder.aembed_query(query)
        count = self.collection.count()
        n_results = min(top_k * 2, count)
        if count == 0:
            return []

        # 预分词（纯 CPU，不阻塞事件循环即可）
        query_tokens = self._tokenize(query)

        # Step 1+2: 向量检索 ‖ BM25 检索（并行，无依赖）
        async def _vector_search():
            return await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                ),
            )

        async def _bm25_search():
            try:
                indices, scores = current_bm25.retrieve(
                    [query_tokens], k=n_results
                )
                if indices.size > 0:
                    valid_indices = indices[0][scores[0] > 0]
                    return [
                        current_ids[i]
                        for i in valid_indices
                        if i < len(current_ids)
                    ]
            except Exception as e:
                logger.error("BM25S retrieval error: %s", e)
            return []

        vector_results, bm25_ids = await asyncio.gather(
            _vector_search(), _bm25_search()
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
        truncated = final_results[:final_top_k]

        # Step 5: Parent-Child 上下文扩展 — 用 Redis 中的父文档替换子片段
        return await self._enrich_with_parent_context(truncated)

    # ========================================================================
    # Parent-Child 上下文扩展
    # ========================================================================

    async def _enrich_with_parent_context(
        self, docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """用 Redis 中的父文档全文替换子片段 text，并去重合并同一 parent 的 chunks。

        流程：
        1. 收集所有 doc 的 parent_id → 去重
        2. 批量从 Redis 拉取父文档全文
        3. 同一 parent_id 的多个 child chunk 合并为一条（保留最高分）
        4. 父文档全文替换 text；若 Redis 未命中，保留原 child text
        5. child_text 字段保留原始片段供溯源
        """
        if not docs:
            return docs

        # 1. 收集唯一的 parent_id
        parent_ids: Dict[str, list[int]] = {}  # parent_id → [doc 索引列表]
        for i, doc in enumerate(docs):
            meta = doc.get("metadata", {}) or {}
            pid = meta.get("parent_id")
            if pid:
                parent_ids.setdefault(pid, []).append(i)

        if not parent_ids:
            logger.debug("[ParentEnrich] No parent_id found in %d docs, skip.", len(docs))
            return docs

        # 2. 批量从 Redis 拉取
        unique_pids = list(parent_ids.keys())
        logger.info("[ParentEnrich] Fetching %d parent contexts from Redis...", len(unique_pids))
        parent_texts = redis_store.batch_get(unique_pids)

        hit_count = sum(1 for v in parent_texts.values() if v)
        logger.info(
            "[ParentEnrich] Redis hit: %d/%d (%.0f%%)",
            hit_count, len(unique_pids),
            (hit_count / len(unique_pids) * 100) if unique_pids else 0,
        )

        # 3. 合并同一 parent 的 chunks → 一条富文档
        enriched: List[Dict[str, Any]] = []
        seen_pids: set[str] = set()

        for doc in docs:
            meta = doc.get("metadata", {}) or {}
            pid = meta.get("parent_id")

            if not pid:
                # 无 parent_id 的文档原样保留
                enriched.append(doc)
                continue

            if pid in seen_pids:
                # 同一 parent 已处理过，跳过（去重合并）
                continue
            seen_pids.add(pid)

            parent_text = parent_texts.get(pid)
            if parent_text:
                # 找到同一 parent 中分数最高的 child
                sibling_indices = parent_ids[pid]
                best_idx = max(
                    sibling_indices,
                    key=lambda j: docs[j].get("rerank_score", docs[j].get("rrf_score", 0)),
                )
                best_doc = docs[best_idx]

                enriched.append({
                    **best_doc,
                    "text": parent_text,                        # ← 替换为父文档全文
                    "child_text": best_doc.get("text", ""),      # 保留原始子片段
                    "sibling_count": len(sibling_indices),       # 同 parent 合并了几个子片段
                })
            else:
                # Redis 未命中 → 保留原 child text，不做合并（可能多个 child 来自不同 parent）
                sibling_indices = parent_ids[pid]
                for idx in sibling_indices:
                    enriched.append({**docs[idx], "child_text": docs[idx].get("text", "")})

        logger.info(
            "[ParentEnrich] %d docs → %d enriched (merged %d parents)",
            len(docs), len(enriched), len(seen_pids),
        )

        return enriched
