"""Hybrid retrieval: Qdrant semantic search + keyword-overlap re-rank + metadata filter.

Qdrant alone doesn't give us BM25-style lexical scoring, so "hybrid" here is
a pragmatic fusion: semantic similarity from Qdrant, blended with a simple
word-overlap score against the retrieved chunk text, weighted by
`retrieval.hybrid_alpha`. Good enough to catch exact-term matches (e.g. a
specific AWS service name) that a pure embedding match might rank lower.
"""

from __future__ import annotations

import re

from meeting_copilot.config import RetrievalConfig, get_config
from meeting_copilot.knowledge.embeddings import LocalEmbedder
from meeting_copilot.pipeline.events import DetectedQuestion, RetrievedChunk, RetrievedContext
from meeting_copilot.retrieval.qdrant_store import QdrantStore, ScoredChunk

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _keyword_overlap_score(query_terms: set[str], chunk_text: str) -> float:
    chunk_terms = _tokenize(chunk_text)
    if not query_terms or not chunk_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


class HybridSearcher:
    def __init__(
        self,
        qdrant: QdrantStore | None = None,
        embedder: LocalEmbedder | None = None,
        config: RetrievalConfig | None = None,
    ):
        self._qdrant = qdrant or QdrantStore()
        self._embedder = embedder or LocalEmbedder()
        self._cfg = config or get_config().retrieval

    async def retrieve(
        self, question: DetectedQuestion, topic_filter: str | None = None
    ) -> RetrievedContext:
        query_text = question.transcript.text
        vector = await self._embedder.embed(query_text)
        hits: list[ScoredChunk] = self._qdrant.search(
            vector, top_k=self._cfg.top_k_vector, topic_filter=topic_filter
        )

        query_terms = _tokenize(query_text)
        fused = [
            (
                hit,
                self._cfg.hybrid_alpha * hit.score
                + (1 - self._cfg.hybrid_alpha) * _keyword_overlap_score(query_terms, hit.text),
            )
            for hit in hits
        ]
        fused.sort(key=lambda pair: pair[1], reverse=True)
        top = fused[: self._cfg.top_k_final]

        chunks = [
            RetrievedChunk(text=hit.text, source=hit.source, topic=hit.topic, score=score)
            for hit, score in top
        ]
        return RetrievedContext(question=question, chunks=chunks)
