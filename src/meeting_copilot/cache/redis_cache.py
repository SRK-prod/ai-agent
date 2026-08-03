"""Redis-backed caches: embedding cache, LLM response cache, STT dedup.

All keys are namespaced under `meeting_copilot:` so the DB can be shared
safely with other tools if needed.
"""

from __future__ import annotations

import hashlib
import json
from typing import cast

import redis.asyncio as redis

from meeting_copilot.config import get_config

_NAMESPACE = "meeting_copilot"


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class RedisCache:
    def __init__(self, client: redis.Redis | None = None):
        cfg = get_config()
        self._client = client or redis.from_url(cfg.secrets.redis_url, decode_responses=True)
        self._cfg = cfg

    async def close(self) -> None:
        await self._client.aclose()

    async def ping(self) -> bool:
        return await self._client.ping()

    # --- embedding cache ---

    def _embedding_key(self, model: str, text: str) -> str:
        return f"{_NAMESPACE}:embedding:{model}:{_hash_key(text)}"

    async def get_embedding(self, model: str, text: str) -> list[float] | None:
        raw = await self._client.get(self._embedding_key(model, text))
        return json.loads(raw) if raw else None

    async def set_embedding(self, model: str, text: str, vector: list[float]) -> None:
        await self._client.set(
            self._embedding_key(model, text),
            json.dumps(vector),
            ex=self._cfg.cache.embedding_ttl_seconds,
        )

    # --- LLM response cache (keyed on question + retrieved-context hash) ---

    def _llm_key(self, question: str, context_chunks: list[str]) -> str:
        return f"{_NAMESPACE}:llm:{_hash_key(question, *context_chunks)}"

    async def get_llm_response(self, question: str, context_chunks: list[str]) -> str | None:
        # client is always constructed with decode_responses=True (see __init__), so this is str
        value = await self._client.get(self._llm_key(question, context_chunks))
        return cast(str | None, value)

    async def set_llm_response(
        self, question: str, context_chunks: list[str], response: str
    ) -> None:
        await self._client.set(
            self._llm_key(question, context_chunks),
            response,
            ex=self._cfg.cache.llm_response_ttl_seconds,
        )

    # --- STT dedup (avoid re-processing an identical utterance seen moments ago) ---

    def _stt_dedup_key(self, speaker_id: str, text: str) -> str:
        return f"{_NAMESPACE}:stt_dedup:{speaker_id}:{_hash_key(text)}"

    async def seen_recently(self, speaker_id: str, text: str, ttl_seconds: int = 10) -> bool:
        """Returns True if this exact utterance was already processed within ttl_seconds."""
        key = self._stt_dedup_key(speaker_id, text)
        was_set = await self._client.set(key, "1", ex=ttl_seconds, nx=True)
        return was_set is None
