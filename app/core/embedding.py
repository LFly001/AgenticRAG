"""向量嵌入服务 — 本地 BGE-M3 模型推理。"""
import asyncio
from typing import List

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridEmbeddings(Embeddings):
    """BGE-M3 本地嵌入，推理放入线程池避免阻塞事件循环。"""

    def __init__(self):
        model_path = settings.LOCAL_EMBEDDING_PATH
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {'torch_dtype': torch.float16} if device == 'cuda' else {}

        logger.info("Loading BGE-M3 from %s on %s", model_path, device)
        self.model = SentenceTransformer(
            model_path, device=device, model_kwargs=model_kwargs,
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量生成向量（线程池执行）。"""
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            ),
        )
        return np.nan_to_num(embeddings).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        """异步生成单条查询向量。"""
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode([text], convert_to_numpy=True)[0],
        )
        return np.nan_to_num(embedding).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Use aembed_documents")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("Use aembed_query")
