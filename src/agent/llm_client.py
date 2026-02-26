from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


_SESSION = requests.Session()


@dataclass
class LLMConfig:
    url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model: str = "local"
    timeout_s: int = 60
    temperature: float = 0.2
    max_tokens: int = 256


def ask_llm(
    cfg: LLMConfig,
    system_prompt: str,
    user_text: str,
    few_shot_messages: list[dict[str, str]] | None = None,
) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_text})

    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    # Reuse a persistent HTTP session to reduce per-request latency.
    r = _SESSION.post(cfg.url, json=payload, timeout=max(5, cfg.timeout_s))
    r.raise_for_status()
    data = r.json()
    if "choices" not in data or not data["choices"]:
        raise ValueError(f"Unexpected LLM response: {data}")
    return str(data["choices"][0]["message"]["content"])