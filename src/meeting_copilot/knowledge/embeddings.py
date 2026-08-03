"""Local, free text embeddings via sentence-transformers -- no API key or billing.

Runs entirely on-device (CPU), swapped in for a hosted embeddings API (e.g.
OpenAI's text-embedding-3-large) per user preference: no external account,
no billing, and the whole retrieval path (see retrieval/hybrid_search.py)
stays fully offline. Search quality is a notch below a large hosted model,
but adequate for meeting-note retrieval.

Used only by pre-meeting ingestion (knowledge/ingestion.py) and by retrieval
to embed the live question -- never for arbitrary internet lookups.
"""

from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from meeting_copilot.cache.redis_cache import RedisCache
from meeting_copilot.config import KnowledgeConfig, get_config

# Loading a SentenceTransformer is expensive (reads model weights from disk);
# share one instance per model name across LocalEmbedder instances.
_model_cache: dict[str, SentenceTransformer] = {}


def _load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class LocalEmbedder:
    def __init__(self, config: KnowledgeConfig | None = None, cache: RedisCache | None = None):
        self._cfg = config or get_config().knowledge
        self._cache = cache or RedisCache()
        self._model = _load_model(self._cfg.embedding_model)
        # Warm up: the first encode() pays lazy-init costs (measured ~0.5s extra on the
        # first live question) -- absorb that at construction/startup instead.
        self._encode_one("warm-up")

    def _encode_one(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def embed(self, text: str) -> list[float]:
        cached = await self._cache.get_embedding(self._cfg.embedding_model, text)
        if cached is not None:
            return cached
        # SentenceTransformer.encode() is CPU-bound and blocking; run off the event loop.
        vector = await asyncio.to_thread(self._encode_one, text)
        await self._cache.set_embedding(self._cfg.embedding_model, text, vector)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = await self._cache.get_embedding(self._cfg.embedding_model, text)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            vectors = await asyncio.to_thread(self._encode_batch, miss_texts)
            for idx, text, vector in zip(miss_indices, miss_texts, vectors, strict=True):
                results[idx] = vector
                await self._cache.set_embedding(self._cfg.embedding_model, text, vector)

        return results  # type: ignore[return-value]
