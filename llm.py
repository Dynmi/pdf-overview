"""Minimal async OpenRouter client."""
import json
import os
from typing import AsyncIterator

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = httpx.Timeout(300.0, connect=15.0)

# Shared pooled client: reuses TCP+TLS across every call for the process, and
# multiplexes many concurrent requests over a single HTTP/2 connection. This
# alone saves ~150ms × N_chunks of handshake time on the map fan-out.
_CLIENT: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        limits = httpx.Limits(
            max_keepalive_connections=int(os.getenv("LLM_MAX_KEEPALIVE", "32")),
            max_connections=int(os.getenv("LLM_MAX_CONNECTIONS", "64")),
            keepalive_expiry=60.0,
        )
        try:
            _CLIENT = httpx.AsyncClient(timeout=TIMEOUT, limits=limits, http2=True)
        except ImportError:
            # h2 not installed — fall back to HTTP/1.1 with the same pool.
            _CLIENT = httpx.AsyncClient(timeout=TIMEOUT, limits=limits)
    return _CLIENT


def cached_system(text: str) -> list[dict]:
    """Build a system-message content array that signals a cache breakpoint to
    providers that honor OpenRouter's `cache_control` passthrough (Anthropic,
    some others). On models that ignore it (Gemini), the structure still parses
    as plain text and nothing breaks. For Anthropic Claude specifically, every
    call after the first hits the cache for this prefix — reducing input-token
    cost by up to ~90% across repeated fan-out."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


async def warmup() -> None:
    """Open a TCP+TLS (and HTTP/2) connection to OpenRouter and leave it in the
    pool, so the first real chat request doesn't pay handshake cost (~150ms)."""
    try:
        await _get_client().head(
            "https://openrouter.ai/api/v1/models",
            timeout=httpx.Timeout(5.0, connect=3.0),
        )
    except Exception:
        # Best-effort; failure here doesn't affect correctness.
        pass


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
    r = await _get_client().post(OPENROUTER_URL, headers=_headers(), json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def chat_stream(model: str, messages: list[dict], **kwargs) -> AsyncIterator[str]:
    payload = {"model": model, "messages": messages, "stream": True, **kwargs}
    async with _get_client().stream(
        "POST", OPENROUTER_URL, headers=_headers(), json=payload
    ) as r:
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
