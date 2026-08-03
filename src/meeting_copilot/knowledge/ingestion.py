"""Pre-meeting knowledge base builder: topics.yaml -> chunks -> embeddings -> Qdrant.

Runs offline, ahead of any meeting (`make ingest`). This is the ONLY place
Claude's general knowledge is used to "research" a topic -- once a meeting
starts, retrieval only ever searches what's already in Qdrant (see
docs/architecture.md, "never trigger internet searches" requirement).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from meeting_copilot.config import get_config
from meeting_copilot.knowledge.chunking import chunk_text
from meeting_copilot.knowledge.embeddings import LocalEmbedder
from meeting_copilot.knowledge.store import KnowledgeStore
from meeting_copilot.llm.claude_client import ClaudeClient
from meeting_copilot.paths import PROJECT_ROOT, TOPICS_FILE
from meeting_copilot.retrieval.qdrant_store import KnowledgeChunkPayload, QdrantStore
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


@dataclass
class TopicSpec:
    name: str
    sources: list[str] = field(default_factory=list)
    research_brief: str | None = None


def load_topics(path: Path | None = None) -> list[TopicSpec]:
    with open(path or TOPICS_FILE) as f:
        raw = yaml.safe_load(f) or {}
    return [TopicSpec(**t) for t in raw.get("topics", [])]


def _read_sources(sources: list[str]) -> list[tuple[str, str]]:
    """Returns [(source_label, text), ...] from local files/dirs listed for a topic."""
    docs: list[tuple[str, str]] = []
    for src in sources:
        path = Path(src)
        if not path.is_absolute():
            path = PROJECT_ROOT / src
        if not path.exists():
            logger.warning(f"Knowledge source not found, skipping: {path}")
            continue
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.suffix.lower() in (".md", ".txt", ".rst"):
                    docs.append((str(file), file.read_text(errors="ignore")))
        else:
            docs.append((str(path), path.read_text(errors="ignore")))
    return docs


async def _research_notes(topic: TopicSpec, claude: ClaudeClient) -> str | None:
    if not topic.research_brief:
        return None
    prompt = (
        f"Write concise, technically accurate reference notes on '{topic.name}' for a "
        f"Principal/Staff Engineer to skim before a meeting. Brief: {topic.research_brief}\n"
        "Cover key concepts, common pitfalls, and tradeoffs. Plain text, no preamble."
    )
    return await claude.complete(prompt)


async def ingest_topic(
    topic: TopicSpec,
    embedder: LocalEmbedder,
    qdrant: QdrantStore,
    knowledge_store: KnowledgeStore,
    claude: ClaudeClient | None = None,
) -> int:
    job_id = knowledge_store.start_job(topic.name)
    try:
        docs = _read_sources(topic.sources)
        if claude is not None:
            notes = await _research_notes(topic, claude)
            if notes:
                docs.append((f"claude-research:{topic.name}", notes))

        if not docs:
            logger.warning(f"No sources or research notes for topic '{topic.name}', skipping.")
            knowledge_store.complete_job(job_id, source_count=0, chunk_count=0)
            return 0

        cfg = get_config().knowledge
        all_chunks: list[str] = []
        chunk_sources: list[str] = []
        for source_label, text in docs:
            for chunk in chunk_text(text, cfg.chunk_tokens, cfg.chunk_overlap_tokens):
                all_chunks.append(chunk)
                chunk_sources.append(source_label)

        vectors = await embedder.embed_batch(all_chunks)
        payloads = [
            KnowledgeChunkPayload(
                text=chunk, topic=topic.name, source=source, chunk_id=f"{topic.name}-{i}"
            )
            for i, (chunk, source) in enumerate(zip(all_chunks, chunk_sources, strict=True))
        ]
        qdrant.upsert_chunks(payloads, vectors)

        knowledge_store.complete_job(job_id, source_count=len(docs), chunk_count=len(all_chunks))
        logger.info(f"Ingested topic '{topic.name}': {len(docs)} sources, {len(all_chunks)} chunks")
        return len(all_chunks)
    except Exception as exc:
        knowledge_store.fail_job(job_id, str(exc))
        raise


async def ingest_all(topics_path: Path | None = None, use_claude_research: bool = True) -> int:
    topics = load_topics(topics_path)
    embedder = LocalEmbedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    knowledge_store = KnowledgeStore()
    claude = None
    if use_claude_research:
        # llm.max_tokens is sized for short live meeting answers -- research notes are the
        # opposite: long, thorough reference docs. Override for this offline job only.
        research_cfg = get_config().llm.model_copy(update={"max_tokens": 4096})
        claude = ClaudeClient(research_cfg)

    total_chunks = 0
    for topic in topics:
        total_chunks += await ingest_topic(topic, embedder, qdrant, knowledge_store, claude)
    return total_chunks


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build the pre-meeting knowledge base in Qdrant")
    parser.add_argument("--topics", type=Path, default=TOPICS_FILE)
    parser.add_argument(
        "--no-research", action="store_true", help="Skip Claude research notes, sources only"
    )
    args = parser.parse_args()

    import asyncio

    total = asyncio.run(ingest_all(args.topics, use_claude_research=not args.no_research))
    logger.info(f"Ingestion complete: {total} chunks stored")


if __name__ == "__main__":
    _cli()
