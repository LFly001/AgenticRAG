"""共享 Redis 连接 — 单例客户端，供 ConversationStore 和 RedisParentStore 共用。

避免每个 Store 各自创建独立连接池，减少 TCP 连接和内存开销。
"""

from __future__ import annotations

import redis
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis | None:
    """返回共享的 Redis 客户端（单例）。

    连接失败时返回 None 而非抛异常，调用方自行降级处理。
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    try:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        _redis_client.ping()
        logger.info(
            "Shared Redis connected at %s:%d", settings.REDIS_HOST, settings.REDIS_PORT
        )
    except Exception as e:
        logger.warning("Shared Redis unavailable: %s. Running in degraded mode.", e)
        _redis_client = None

    return _redis_client
