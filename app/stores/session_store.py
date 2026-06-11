"""会话记忆存储 — 基于 Redis LIST 的多轮对话上下文管理。

Key 格式:  conv:{session_id}
Value:     Redis LIST，每条元素为 JSON 字符串 [{role, content, timestamp}, ...]
TTL:       24 小时（可配置）

设计要点：
- 使用 RPUSH + LTRIM 原子操作，消除 read-modify-write 竞态
- 共享 Redis 连接（app.stores.redis_client），避免多连接池浪费
"""

import json
import time
from typing import List, Dict

from app.config import settings
from app.stores.redis_client import get_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationStore:
    """Redis 对话记忆存储 — 基于 LIST，原子操作无竞态。

    用法：
        store = ConversationStore()
        store.append(session_id, "user", "什么是OKR？")
        store.append(session_id, "assistant", "OKR是...")
        history = store.get_history(session_id)
    """

    def __init__(self):
        self._client = get_redis_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, session_id: str) -> str:
        return f"conv:{session_id}"

    # ---- public API ----

    def append(self, session_id: str, role: str, content: str) -> None:
        """原子追加一条对话记录 — 截断 + RPUSH + LTRIM + EXPIRE，无竞态。"""
        if not self._client or not session_id:
            return
        try:
            max_chars = settings.MAX_MESSAGE_CHARS
            trimmed = content if len(content) <= max_chars else content[:max_chars] + "..."

            entry = {
                "role": role,
                "content": trimmed,
                "timestamp": time.time(),
            }
            key = self._key(session_id)
            max_entries = settings.MAX_HISTORY_TURNS * 2
            pipe = self._client.pipeline()
            pipe.rpush(key, json.dumps(entry, ensure_ascii=False))
            pipe.ltrim(key, -max_entries, -1)
            pipe.expire(key, settings.CONV_TTL)
            pipe.execute()
        except Exception as e:
            logger.warning("ConversationStore append failed: %s", e)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取对话历史（不含 timestamp，LRANGE 全量读取）。"""
        if not self._client or not session_id:
            return []
        try:
            raw_entries = self._client.lrange(self._key(session_id), 0, -1)
            if not raw_entries:
                return []
            history = []
            for raw in raw_entries:
                try:
                    h = json.loads(raw)
                    history.append({"role": h["role"], "content": h["content"]})
                except (json.JSONDecodeError, KeyError):
                    continue
            return history
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
