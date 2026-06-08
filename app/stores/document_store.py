"""父文档上下文存储 — 基于 Redis 的 Parent-Child 父块缓存，带降级处理。"""
import redis
from typing import Optional, Dict
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisParentStore:
    """
    基于 Redis 的父文档上下文存储
    Key 格式: rag:parent:{parent_id}
    Value: 完整的文本内容
    """
    def __init__(self):
        # 从 settings 获取 Redis 配置，如果没有则使用默认本地配置
        self.redis_host = getattr(settings, 'REDIS_HOST', 'localhost')
        self.redis_port = getattr(settings, 'REDIS_PORT', 6379)
        self.redis_db = getattr(settings, 'REDIS_DB', 0)
        self.redis_password = getattr(settings, 'REDIS_PASSWORD', None)

        self.client = None
        try:
            self.client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,  # 自动解码 bytes 为 str
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 测试连接
            self.client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            # 降低日志级别为 WARNING，因为系统设计了降级方案
            logger.warning(
                "Redis connection failed: %s. "
                "Parent context storage disabled. Using fallback mode.",
                e,
            )
            self.client = None

    def save_parent_context(self, parent_id: str, text: str, expire_seconds: int = None):
        """
        保存父上下文到 Redis
        """
        if not self.client:
            return
        try:
            key = f"rag:parent:{parent_id}"
            pipe = self.client.pipeline()
            pipe.set(key, text)
            if expire_seconds:
                pipe.expire(key, expire_seconds)
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed to save parent context to Redis for {parent_id}: {e}")

    def get_parent_context(self, parent_id: str) -> Optional[str]:
        """
        从 Redis 获取父上下文
        """
        if not self.client:
            return None
        try:
            key = f"rag:parent:{parent_id}"
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Failed to get parent context from Redis for {parent_id}: {e}")
            return None

    def batch_save(self, items: Dict[str, str], expire_seconds: int = None):
        """
        批量保存父上下文
        """
        if not self.client or not items:
            return
        try:
            pipe = self.client.pipeline()
            for pid, text in items.items():
                key = f"rag:parent:{pid}"
                pipe.set(key, text)
                if expire_seconds:
                    pipe.expire(key, expire_seconds)
            pipe.execute()
            logger.debug(f"Batch saved {len(items)} parent contexts to Redis.")
        except Exception as e:
            logger.error(f"Failed to batch save parent contexts: {e}")

    def delete_parent_context(self, parent_id: str):
        """删除指定的父上下文"""
        if not self.client:
            return
        try:
            key = f"rag:parent:{parent_id}"
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete parent context {parent_id}: {e}")


# 单例
redis_store = RedisParentStore()
