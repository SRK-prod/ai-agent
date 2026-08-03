"""Pre-generated Q&A bank: anticipated interview questions with full answers.

Built offline by scripts/build_qa_bank.py (questions + structured answers,
grounded in the candidate's profile and the knowledge base). At meeting time,
the asked question is embedded and matched against banked questions -- a hit
above `qa_bank.similarity_threshold` serves the stored answer instantly
(<1s, no LLM call) instead of the ~4-5s live generation path.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from meeting_copilot.config import get_config


@dataclass
class BankedAnswer:
    question: str
    answer: str  # raw answer text including the trailing CONFIDENCE line
    topic: str
    score: float


def _point_id(question: str) -> str:
    # deterministic so rebuilding the bank overwrites rather than duplicates
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qa-bank:{question.lower().strip()}"))


class QaBankStore:
    def __init__(self, client: QdrantClient | None = None):
        cfg = get_config()
        self._cfg = cfg.qa_bank
        self._dim = cfg.knowledge.embedding_dim
        self._client = client or QdrantClient(url=cfg.secrets.qdrant_url)

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._cfg.collection_name):
            self._client.create_collection(
                collection_name=self._cfg.collection_name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def upsert(self, question: str, answer: str, topic: str, vector: list[float]) -> None:
        self._client.upsert(
            collection_name=self._cfg.collection_name,
            points=[
                PointStruct(
                    id=_point_id(question),
                    vector=vector,
                    payload={"question": question, "answer": answer, "topic": topic},
                )
            ],
        )

    def count(self) -> int:
        if not self._client.collection_exists(self._cfg.collection_name):
            return 0
        return self._client.count(self._cfg.collection_name).count

    def lookup(self, vector: list[float]) -> BankedAnswer | None:
        """Best banked match for the asked question, or None below the threshold."""
        if not self._client.collection_exists(self._cfg.collection_name):
            return None
        hits = self._client.query_points(
            collection_name=self._cfg.collection_name, query=vector, limit=1
        ).points
        if not hits or hits[0].score < self._cfg.similarity_threshold:
            return None
        hit = hits[0]
        assert hit.payload is not None
        return BankedAnswer(
            question=hit.payload["question"],
            answer=hit.payload["answer"],
            topic=hit.payload.get("topic", ""),
            score=hit.score,
        )
