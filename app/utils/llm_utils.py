"""LLM 响应解析工具 — 统一处理 JSON 提取、markdown 代码块剥离等。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """从 LLM 响应中提取 JSON 对象。

    处理以下格式：
    - 纯 JSON:        '{"key": "value"}'
    - markdown 包裹:  '```json\\n{"key": "value"}\\n```'
    - 无语言标记包裹:  '```\\n{"key": "value"}\\n```'
    - 前后有空白字符

    Raises:
        json.JSONDecodeError: 解析失败时抛出。
    """
    text = raw.strip()

    # 去 markdown 代码块包裹
    # 匹配开头的 ```json 或 ``` 并移除
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    # 匹配结尾的 ``` 并移除
    text = re.sub(r'\n?```\s*$', '', text)

    return json.loads(text)


def safe_parse_llm_json(raw: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """安全解析 LLM JSON，解析失败返回 default。

    Args:
        raw: LLM 原始响应文本。
        default: 解析失败时的默认值（若为 None 则返回空 dict）。

    Returns:
        解析出的 dict，或 default。
    """
    try:
        return parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError):
        return default if default is not None else {}
