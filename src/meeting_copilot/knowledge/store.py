"""SQLite ingestion-job log (the spec's "Knowledge Store: SQLite").

Qdrant holds the actual chunk vectors + payload; this is just the audit
trail of what got ingested, when, and whether it succeeded -- so
`make ingest` runs are debuggable without digging through Qdrant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)

from meeting_copilot.paths import DATA_DIR

_metadata = MetaData()

_ingestion_jobs = Table(
    "ingestion_jobs",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("topic", String, nullable=False),
    Column("source_count", Integer, nullable=False, default=0),
    Column("chunk_count", Integer, nullable=False, default=0),
    Column("status", String, nullable=False),  # running | completed | failed
    Column("error", Text, nullable=True),
    Column("started_at", DateTime, nullable=False),
    Column("completed_at", DateTime, nullable=True),
)


class KnowledgeStore:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or DATA_DIR / "knowledge.sqlite3")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{path}")
        _metadata.create_all(self._engine)

    def start_job(self, topic: str) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(
                insert(_ingestion_jobs).values(
                    topic=topic,
                    source_count=0,
                    chunk_count=0,
                    status="running",
                    started_at=datetime.now(UTC),
                )
            )
            assert result.inserted_primary_key is not None
            return result.inserted_primary_key[0]

    def complete_job(self, job_id: int, source_count: int, chunk_count: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(_ingestion_jobs)
                .where(_ingestion_jobs.c.id == job_id)
                .values(
                    status="completed",
                    source_count=source_count,
                    chunk_count=chunk_count,
                    completed_at=datetime.now(UTC),
                )
            )

    def fail_job(self, job_id: int, error: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(_ingestion_jobs)
                .where(_ingestion_jobs.c.id == job_id)
                .values(status="failed", error=error, completed_at=datetime.now(UTC))
            )

    def recent_jobs(self, limit: int = 20) -> list[dict]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(_ingestion_jobs).order_by(_ingestion_jobs.c.id.desc()).limit(limit)
            ).mappings()
            return [dict(row) for row in rows]
