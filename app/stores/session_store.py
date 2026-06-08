"""会话记忆存储 — 基于 Redis 的多轮对话上下文管理。

Key 格式:  conv:{session_id}
Value:     JSON 数组 [{role, content, timestamp}, ...]
TTL:       24 小时（可配置）
"""

import json
import time
from typing import List, Dict, Optional

import redis

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 默认对话保留轮数
MAX_HISTORY_TURNS = 10
# Redis key 过期时间（秒），默认 24 小时
DEFAULT_TTL = 86400


class ConversationStore:
    """Redis 对话记忆存储。

    用法：
        store = ConversationStore()
        store.append(session_id, "user", "什么是OKR？")
        store.append(session_id, "assistant", "OKR是...")
        history = store.get_history(session_id)
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        try:
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._client.ping()
            logger.info(
                "ConversationStore connected to Redis at %s:%d",
                settings.REDIS_HOST, settings.REDIS_PORT,
            )
        except Exception as e:
            logger.warning(
                "ConversationStore: Redis unavailable (%s). "
                "Multi-turn memory disabled.", e,
            )
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, session_id: str) -> str:
        return f"conv:{session_id}"

    # ---- public API ----

    def append(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话记录。"""
        if not self._client or not session_id:
            return
        try:
            entry = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }
            key = self._key(session_id)
            pipe = self._client.pipeline()
            raw = self._client.get(key)
            history: list = json.loads(raw) if raw else []
            history.append(entry)
            # 只保留最近 N 轮（每轮 = user + assistant）
            if len(history) > MAX_HISTORY_TURNS * 2:
                history = history[-(MAX_HISTORY_TURNS * 2):]
            pipe.set(key, json.dumps(history, ensure_ascii=False))
            pipe.expire(key, DEFAULT_TTL)
            pipe.execute()
        except Exception as e:
            logger.warning("ConversationStore append failed: %s", e)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取对话历史（不含 timestamp）。"""
        if not self._client or not session_id:
            return []
        try:
            raw = self._client.get(self._key(session_id))
            if not raw:
                return []
            history = json.loads(raw)
            # 只返回 role + content，去掉 timestamp
            return [
                {"role": h["role"], "content": h["content"]}
                for h in history
            ]
        except Exception as e:
            logger.warning("ConversationStore get_history failed: %s", e)
            return []

    def clear(self, session_id: str) -> None:
        """清除指定会话的对话历史。"""
        if not self._client or not session_id:
            return
        try:
            self._client.delete(self._key(session_id))
        except Exception as e:
            logger.warning("ConversationStore clear failed: %s", e)


# 单例
conv_store = ConversationStore()
