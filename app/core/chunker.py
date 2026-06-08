import hashlib
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger
from app.stores.document_store import redis_store  # 引入 Redis 存储


logger = get_logger(__name__)


class SmartChunker:
    """
    智能切片器：
    1. 区分表格和普通文本。
    2. 实现父子切片逻辑：
       - Child Chunk: 小片段，用于向量检索（高精度）。
       - Parent Context: 大片段，存储在 Redis 中，通过 ID 引用。
    """

    def __init__(self):
        # 普通文本切片器 (Child Splitter)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,       # 例如 500
            chunk_overlap=settings.CHUNK_OVERLAP,  # 例如 50
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _generate_stable_id(self, text: str, source: str, index: int) -> str:
        """生成基于内容的稳定 ID"""
        content_hash = hashlib.md5(f"{source}_{index}_{text[:200]}".encode('utf-8')).hexdigest()
        return f"chunk_{content_hash}"

    def chunk_documents(self, parsed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        输入：解析后的文档列表 (List[Dict])
        输出：适合存入向量库的切片列表 (List[Dict])

        注意：父文本现在存储在 Redis 中，metadata 中仅保留 parent_id
        """
        all_chunks = []
        parent_contexts_to_save = {}

        for doc_idx, doc in enumerate(parsed_docs):
            text = doc["text"]
            metadata = doc["metadata"]
            source = metadata.get("source", "unknown")
            page = metadata.get("page_number", 0)

            # 生成稳定的 Parent ID
            parent_id = self._generate_stable_id(text, source, doc_idx)

            # 【关键优化】将父文本暂存，稍后批量写入 Redis
            parent_contexts_to_save[parent_id] = text

            # --- 策略 1: 表格处理 ---
            if metadata.get("is_table", False):
                chunk_obj = {
                    "id": f"{parent_id}_table",
                    "text": text,
                    "metadata": {
                        **metadata,
                        "chunk_type": "table",
                        "parent_id": parent_id,  # 仅保留 ID
                        "source_file": source,
                        "page_number": page
                    }
                }
                all_chunks.append(chunk_obj)

            # --- 策略 2: 普通文本处理 (父子切片) ---
            else:
                # 1. 生成子切片 (用于检索)
                child_splits = self.child_splitter.split_text(text)

                if not child_splits:
                    continue

                for i, split in enumerate(child_splits):
                    child_id = f"{parent_id}_seg{i}"

                    chunk_obj = {
                        "id": child_id,
                        "text": split,  # 小片段：用于计算向量相似度
                        "metadata": {
                            **metadata,
                            "chunk_type": "text",
                            "parent_id": parent_id,  # 仅保留 ID
                            "source_file": source,
                            "page_number": page,
                            "segment_index": i
                        }
                    }
                    all_chunks.append(chunk_obj)

        # 【关键优化】批量保存父上下文到 Redis
        if parent_contexts_to_save:
            redis_store.batch_save(parent_contexts_to_save)
            logger.info(f"Saved {len(parent_contexts_to_save)} parent contexts to Redis.")

        logger.info(f"Generated {len(all_chunks)} chunks from {len(parsed_docs)} elements.")
        return all_chunks
