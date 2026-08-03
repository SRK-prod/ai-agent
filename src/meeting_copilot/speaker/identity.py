"""Enrolled-voice storage (SQLite) + online speaker clustering.

Storage is a single small table: the enrolled "me" embedding produced once by
scripts/enroll_voice.py. During a meeting, `SpeakerIdentity` compares every
diarized segment's embedding against that vector to decide "ignore (me)" vs.
process, and does simple nearest-centroid online clustering to label the
other participants Speaker A/B/C without needing to know their identity
ahead of time.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from meeting_copilot.config import SpeakerConfig, get_config
from meeting_copilot.paths import PROJECT_ROOT
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

_SPEAKER_LABELS = [chr(ord("A") + i) for i in range(26)]  # Speaker A, B, C, ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


class EnrollmentStore:
    """Single-table SQLite store for the enrolled "me" voice embedding."""

    def __init__(self, db_path: str | Path | None = None):
        cfg = get_config().speaker
        self._path = Path(db_path or PROJECT_ROOT / cfg.enrollment_db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrollment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(self, embedding: np.ndarray, embedding_model: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM enrollment")  # single active enrollment
            conn.execute(
                "INSERT INTO enrollment (embedding_json, embedding_model, created_at) "
                "VALUES (?, ?, ?)",
                (
                    json.dumps(embedding.flatten().tolist()),
                    embedding_model,
                    datetime.now(UTC).isoformat(),
                ),
            )
        logger.info(f"Saved voice enrollment ({embedding_model}) to {self._path}")

    def load(self) -> np.ndarray | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT embedding_json FROM enrollment ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return np.array(json.loads(row[0]), dtype=np.float32)


class SpeakerIdentity:
    """Online speaker labeling: "me" vs. nearest-centroid clustering for others."""

    def __init__(self, config: SpeakerConfig | None = None, store: EnrollmentStore | None = None):
        self._cfg = config or get_config().speaker
        self._store = store or EnrollmentStore()
        self._me_embedding = self._store.load()
        if self._me_embedding is None:
            logger.warning(
                "No enrolled voice found -- every speaker will be treated as a participant "
                "until you run `make enroll` (scripts/enroll_voice.py)."
            )
        # label -> (centroid embedding, sample count)
        self._known_speakers: dict[str, tuple[np.ndarray, int]] = {}

    def _next_label(self) -> str:
        for label in _SPEAKER_LABELS:
            if label not in self._known_speakers:
                return label
        # wrap around and reuse the oldest label if we somehow exceed 26 speakers
        return _SPEAKER_LABELS[len(self._known_speakers) % len(_SPEAKER_LABELS)]

    def assign_speaker(self, embedding: np.ndarray) -> tuple[str, bool, float]:
        """Returns (speaker_id, is_me, similarity_to_me)."""
        similarity_to_me = (
            cosine_similarity(embedding, self._me_embedding)
            if self._me_embedding is not None
            else 0.0
        )
        if similarity_to_me >= self._cfg.ignore_similarity_threshold:
            return "me", True, similarity_to_me

        best_label, best_similarity = None, -1.0
        for label, (centroid, _count) in self._known_speakers.items():
            sim = cosine_similarity(embedding, centroid)
            if sim > best_similarity:
                best_label, best_similarity = label, sim

        if best_label is not None and best_similarity >= self._cfg.new_speaker_similarity_threshold:
            centroid, count = self._known_speakers[best_label]
            new_centroid = (centroid * count + embedding) / (count + 1)
            self._known_speakers[best_label] = (new_centroid, count + 1)
            return best_label, False, similarity_to_me

        if len(self._known_speakers) >= self._cfg.max_tracked_speakers:
            label = best_label or self._next_label()
        else:
            label = self._next_label()
        self._known_speakers[label] = (embedding.copy(), 1)
        logger.debug(f"New speaker tracked: {label}")
        return label, False, similarity_to_me
