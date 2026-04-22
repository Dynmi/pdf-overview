"""Minimal async OpenRouter client."""
import json
import os
from typing import AsyncIterator

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = httpx.Timeout(300.0, connect=15.0)


def _headers() -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "PDF Overview",
    }


async def chat(model: str, messages: list[dict], **kwargs) -> str:
    payload = {"model": model, "messages": messages, "stream": False, **kwargs}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(OPENROUTER_URL, headers=_headers(), json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def chat_stream(model: str, messages: list[dict], **kwargs) -> AsyncIterator[str]:
    payload = {"model": model, "messages": messages, "stream": True, **kwargs}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        async with c.stream("POST", OPENROUTER_URL, headers=_headers(), json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
