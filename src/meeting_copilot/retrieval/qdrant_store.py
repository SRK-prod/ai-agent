"""Thin wrapper around qdrant-client for the knowledge-base collection."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from meeting_copilot.config import KnowledgeConfig, get_config


@dataclass
class KnowledgeChunkPayload:
    text: str
    topic: str
    source: str
    chunk_id: str


@dataclass
class ScoredChunk:
    text: str
    topic: str
    source: str
    chunk_id: str
    score: float


def _point_id(chunk_id: str) -> str:
    # deterministic UUID so re-ingesting the same chunk_id overwrites rather than duplicates
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, config: KnowledgeConfig | None = None, client: QdrantClient | None = None):
        self._cfg = config or get_config().knowledge
        self._client = client or QdrantClient(url=get_config().secrets.qdrant_url)

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._cfg.collection_name):
            self._client.create_collection(
                collection_name=self._cfg.collection_name,
                vectors_config=VectorParams(size=self._cfg.embedding_dim, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self, payloads: list[KnowledgeChunkPayload], vectors: list[list[float]]
    ) -> None:
        points = [
            PointStruct(
                id=_point_id(payload.chunk_id),
                vector=vector,
                payload={
                    "text": payload.text,
                    "topic": payload.topic,
                    "source": payload.source,
                    "chunk_id": payload.chunk_id,
                },
            )
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._cfg.collection_name, points=points)

    def search(
        self, vector: list[float], top_k: int, topic_filter: str | None = None
    ) -> list[ScoredChunk]:
        query_filter = None
        if topic_filter:
            query_filter = Filter(
                must=[FieldCondition(key="topic", match=MatchValue(value=topic_filter))]
            )
        hits = self._client.query_points(
            collection_name=self._cfg.collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
        ).points

        chunks = []
        for hit in hits:
            assert hit.payload is not None, f"point {hit.id} was upserted without a payload"
            chunks.append(
                ScoredChunk(
                    text=hit.payload["text"],
                    topic=hit.payload["topic"],
                    source=hit.payload["source"],
                    chunk_id=hit.payload["chunk_id"],
                    score=hit.score,
                )
            )
        return chunks
