"""文档入库 — BM25 索引构建、ChromaDB 写入、文档删除。

IndexingMixin 需与 HybridRetriever 配合使用：
HybridRetriever 提供 __init__ 中的共享状态（ChromaDB、embedder、BM25 锁等），
IndexingMixin 在此基础上提供入库相关方法。
"""

import asyncio
from typing import List, Dict, Any

import bm25s

from app.stores.document_store import redis_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IndexingMixin:
    """入库方法混入 — 依赖 self.collection / self.embedder / self.bm25_* 等属性。

    这些属性由 HybridRetriever.__init__ 初始化，此处仅声明类型以消除 IDE 警告。
    """

    collection: Any
    embedder: Any
    bm25_retriever: Any
    bm25_corpus_ids: List[str]
    bm25_corpus_texts: List[str]
    bm25_lock: asyncio.Lock

    # ---- metadata ----

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        cleaned = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                key = str(key)
            if value is None:
                cleaned[key] = "N/A"
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                try:
                    cleaned[key] = str(value)
                except Exception:
                    cleaned[key] = "Unserializable Object"
        return cleaned

    # ---- BM25 index (internal) ----

    def _init_bm25s_from_db(self):
        """启动时从 ChromaDB 全量加载并构建 BM25 索引。"""
        try:
            existing_ids = self.collection.get(include=[])['ids']
            if not existing_ids:
                logger.info("No existing documents in DB. BM25 index is empty.")
                self.bm25_retriever = bm25s.BM25()
                self.bm25_corpus_ids = []
                self.bm25_corpus_texts = []
                return

            all_texts = []
            all_ids = []
            batch_size = 1000

            logger.info(
                "Loading %d documents for BM25 initialization...", len(existing_ids)
            )
            for i in range(0, len(existing_ids), batch_size):
                batch_ids = existing_ids[i:i + batch_size]
                batch_data = self.collection.get(
                    ids=batch_ids, include=["documents"]
                )
                all_texts.extend(batch_data['documents'])
                all_ids.extend(batch_ids)

            self._build_bm25s_index_internal(all_texts, all_ids)
            logger.info("Initialized BM25S index with %d documents.", len(all_ids))

        except Exception as e:
            logger.error(
                "Failed to initialize BM25S from DB: %s", e, exc_info=True
            )
            self.bm25_retriever = bm25s.BM25()
            self.bm25_corpus_ids = []
            self.bm25_corpus_texts = []

    def _build_bm25s_index_internal(self, texts: List[str], ids: List[str]):
        """根据 texts + ids 构建 BM25 对象（不加锁，调用方负责线程安全）。"""
        if not texts:
            self.bm25_retriever = bm25s.BM25()
            self.bm25_corpus_ids = []
            self.bm25_corpus_texts = []
            return

        logger.debug("Building BM25S index for %d documents...", len(texts))
        tokenized_texts = [self._tokenize(t) for t in texts]
        retriever = bm25s.BM25()
        retriever.index(tokenized_texts)

        self.bm25_retriever = retriever
        self.bm25_corpus_ids = ids
        self.bm25_corpus_texts = texts
        logger.info("BM25S index built successfully.")

    def _add_to_bm25s_index_incremental(
        self, new_texts: List[str], new_ids: List[str]
    ):
        """增量添加文档到 BM25 索引（小批量重建策略，自动去重）。"""
        if not new_texts:
            return

        # 过滤已存在的 ID，避免 ChromaDB upsert 后 BM25 产生重复
        existing = set(self.bm25_corpus_ids)
        deduped = [
            (tid, txt) for tid, txt in zip(new_ids, new_texts)
            if tid not in existing
        ]
        if not deduped:
            logger.info("BM25 index already up-to-date, no new IDs to add.")
            return

        new_ids_dedup, new_texts_dedup = zip(*deduped)
        all_texts = self.bm25_corpus_texts + list(new_texts_dedup)
        all_ids = self.bm25_corpus_ids + list(new_ids_dedup)

        logger.debug(
            "Incrementally updating BM25 index. Total docs: %d", len(all_texts)
        )
        tokenized_texts = [self._tokenize(t) for t in all_texts]

        new_retriever = bm25s.BM25()
        new_retriever.index(tokenized_texts)

        self.bm25_retriever = new_retriever
        self.bm25_corpus_ids = all_ids
        self.bm25_corpus_texts = all_texts

    def _rebuild_bm25s_after_deletion(self, ids_to_delete: set):
        """删除后重建 BM25 索引：过滤已删除 ID 后重建。"""
        try:
            new_texts = []
            new_ids = []
            for tid, txt in zip(self.bm25_corpus_ids, self.bm25_corpus_texts):
                if tid not in ids_to_delete:
                    new_ids.append(tid)
                    new_texts.append(txt)

            self._build_bm25s_index_internal(new_texts, new_ids)
            logger.info(
                "BM25 index rebuilt after deletion. Remaining docs: %d",
                len(new_ids),
            )
        except Exception as e:
            logger.error("Failed to rebuild BM25S after deletion: %s", e)

    # ---- document summary cache ----

    def _build_doc_summary_from_db(self):
        """启动时从 ChromaDB 全量构建文档摘要缓存（仅执行一次）。"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            metadatas = all_data.get("metadatas", [])

            self._doc_summary.clear()
            for meta in (metadatas or []):
                if isinstance(meta, dict):
                    source = meta.get("source_file", "unknown")
                    self._doc_summary[source] = self._doc_summary.get(source, 0) + 1

            logger.info(
                "Doc summary built: %d documents, %d chunks.",
                len(self._doc_summary),
                sum(self._doc_summary.values()),
            )
        except Exception as e:
            logger.warning("Failed to build doc summary from DB: %s", e)
            self._doc_summary = {}

    # ---- public API ----

    async def list_all_documents(self) -> List[Dict[str, Any]]:
        """列出知识库中所有文档（从内存缓存读取，零延迟）。"""
        result = [
            {"filename": filename, "chunk_count": count}
            for filename, count in sorted(
                self._doc_summary.items(), key=lambda x: x[1], reverse=True
            )
        ]
        logger.info("Listed %d unique documents from cache.", len(result))
        return result

    async def delete_documents_by_source(self, source_file: str) -> int:
        """根据文件名删除相关 Chunks、Redis Parent Contexts，并重建 BM25 索引。
        返回删除的切片数量。
        """
        logger.info("Starting deletion for source: %s", source_file)
        loop = asyncio.get_running_loop()

        try:
            # 1. 从 ChromaDB 获取所有相关 ID
            result = await loop.run_in_executor(None, lambda: self.collection.get(
                where={"source_file": source_file},
                include=["metadatas"],
            ))

            ids_to_delete = set(result['ids'])
            metadatas = result['metadatas']
            deleted_count = len(ids_to_delete)

            if not ids_to_delete:
                logger.info(
                    "No documents found for %s. Nothing to delete.", source_file
                )
                return 0

            # 2. 提取 Parent IDs 并从 Redis 删除
            parent_ids_to_delete = set()
            for meta in metadatas:
                pid = meta.get('parent_id')
                if pid:
                    parent_ids_to_delete.add(pid)

            if parent_ids_to_delete:
                for pid in parent_ids_to_delete:
                    redis_store.delete_parent_context(pid)
                logger.info(
                    "Deleted %d parent contexts from Redis.",
                    len(parent_ids_to_delete),
                )

            # 3. 从 ChromaDB 删除 Chunks
            await loop.run_in_executor(
                None,
                lambda: self.collection.delete(ids=list(ids_to_delete)),
            )
            logger.info(
                "Deleted %d chunks from ChromaDB.", deleted_count
            )

            # 4. 增量重建 BM25S 索引
            async with self.bm25_lock:
                await loop.run_in_executor(
                    None, self._rebuild_bm25s_after_deletion, ids_to_delete
                )

            # 5. 从文档摘要缓存中移除
            self._doc_summary.pop(source_file, None)

            logger.info(
                "Successfully deleted and rebuilt index for %s.", source_file
            )
            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to delete documents for %s: %s",
                source_file, e, exc_info=True,
            )
            raise e

    async def add_documents_to_index(self, chunks: List[Dict[str, Any]]):
        """异步添加文档到 ChromaDB 向量库和 BM25 索引。"""
        if not chunks:
            return

        await self._ensure_models()  # 懒加载触发（若未加载）

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            self._clean_metadata(chunk.get("metadata", {})) for chunk in chunks
        ]

        logger.info("Generating embeddings for %d chunks...", len(texts))
        embeddings = await self.embedder.aembed_documents(texts)

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(None, lambda: self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            ))
        except Exception as e:
            logger.error("Failed to add documents to ChromaDB: %s", e)
            raise e

        # 增量更新 BM25 索引
        async with self.bm25_lock:
            await loop.run_in_executor(
                None, self._add_to_bm25s_index_incremental, texts, ids
            )

        # 更新文档摘要缓存
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source_file", "unknown")
            self._doc_summary[source] = self._doc_summary.get(source, 0) + 1

        logger.info("Successfully indexed %d chunks.", len(chunks))
