"""统一 LLM 工厂 — 按配置缓存 ChatOpenAI 实例，消除 8 个 Agent 中的重复创建。

使用方式：
    from app.core.llm_factory import get_llm

    llm = get_llm(temperature=0, max_tokens=1024, timeout=30)
"""

from __future__ import annotations

from typing import Dict

from langchain_openai import ChatOpenAI

from app.config import settings

# 全局缓存：key = "temperature_maxTokens_timeout" → ChatOpenAI 实例
_llm_cache: Dict[str, ChatOpenAI] = {}


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 30,
) -> ChatOpenAI:
    """获取或创建共享的 ChatOpenAI 实例。

    相同 (temperature, max_tokens, timeout) 参数复用同一个实例，
    底层 httpx 连接池被共享，减少内存和连接开销。
    """
    key = f"{temperature}_{max_tokens}_{timeout}"

    if key not in _llm_cache:
        _llm_cache[key] = ChatOpenAI(
            model_name=settings.LLM_MODEL_NAME,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base=settings.DEEPSEEK_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout,
            max_retries=2,
        )

    return _llm_cache[key]


def clear_llm_cache() -> None:
    """清空 LLM 缓存（主要用于测试）。"""
    _llm_cache.clear()
