"""父文档上下文存储 — 基于 Redis 的 Parent-Child 父块缓存，带降级处理。"""
from typing import Optional, Dict

from app.stores.redis_client import get_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisParentStore:
    """
    基于 Redis 的父文档上下文存储
    Key 格式: rag:parent:{parent_id}
    Value: 完整的文本内容
    """
    def __init__(self):
        self.client = get_redis_client()
        if self.client:
            logger.info("RedisParentStore ready (shared connection).")
        else:
            logger.warning(
                "RedisParentStore: Redis unavailable. "
                "Parent context enrichment disabled."
            )

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

    def batch_get(self, parent_ids: list[str]) -> Dict[str, Optional[str]]:
        """批量获取父上下文，一次 pipeline 减少 RTT。

        Args:
            parent_ids: 去重后的 parent_id 列表。

        Returns:
            {parent_id: text | None} 字典，未命中或异常时 value 为 None。
        """
        if not self.client or not parent_ids:
            return {}
        try:
            pipe = self.client.pipeline()
            for pid in parent_ids:
                pipe.get(f"rag:parent:{pid}")
            values = pipe.execute()
            return {
                pid: (val.decode("utf-8") if isinstance(val, bytes) else val)
                for pid, val in zip(parent_ids, values)
            }
        except Exception as e:
            logger.error("Failed to batch get parent contexts: %s", e)
            return {}

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
