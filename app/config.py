"""应用配置 — 环境变量与模型参数设置。"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # --- LLM ---
    DEEPSEEK_API_KEY: str = Field(..., description="DeepSeek API Key")
    DEEPSEEK_BASE_URL: str = Field("https://api.deepseek.com/v1")
    LLM_MODEL_NAME: str = Field("deepseek-chat")

    # --- Embedding ---
    LOCAL_EMBEDDING_PATH: str = Field(
        "./ml_models/bge-m3",
        description="BGE-M3 模型路径",
    )

    # --- HuggingFace ---
    HF_ENDPOINT: Optional[str] = Field(
        None,
        description="HuggingFace 镜像地址，国内用户可设 https://hf-mirror.com",
    )

    # --- Reranker ---
    RERANKER_MODEL_NAME: str = Field("./ml_models/bge-reranker-v2-m3", description="重排序模型路径")

    # --- Vector DB (ChromaDB) ---
    CHROMA_PERSIST_DIR: str = Field("./chroma_db")
    COLLECTION_NAME: str = Field("enterprise_knowledge")

    # --- Retrieval ---
    TOP_K_RETRIEVAL: int = Field(20)
    TOP_K_RERANK: int = Field(5)
    RRF_K_CONSTANT: int = Field(60)

    # --- Chunking ---
    CHUNK_SIZE: int = Field(256)
    CHUNK_OVERLAP: int = Field(32)

    # --- Redis ---
    REDIS_HOST: str = Field("localhost")
    REDIS_PORT: int = Field(6379)
    REDIS_DB: int = Field(0)
    REDIS_PASSWORD: Optional[str] = Field(None)
    REDIS_MAX_MEMORY: str = Field("512mb", description="Redis 最大内存上限")
    REDIS_MAX_MEMORY_POLICY: str = Field("allkeys-lru", description="内存满时淘汰策略")
    PARENT_CONTEXT_TTL: int = Field(2592000, description="父文档 Redis 过期时间（秒），默认 30 天")
    PARENT_CONTEXT_MAX_CHARS: int = Field(8000, description="单个父文档最大字符数，超出截断")

    # --- Conversation ---
    MAX_HISTORY_TURNS: int = Field(10, description="会话保留最大轮数（每轮=user+assistant）")
    MAX_MESSAGE_CHARS: int = Field(2000, description="单条消息存入 Redis 的最大字符数，超出截断")
    CONV_TTL: int = Field(86400, description="会话 Redis key 过期时间（秒），默认 24 小时")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    raise e
