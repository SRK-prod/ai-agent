"""Throughput check for chunking a large ingested document -- ingestion runs
offline so this isn't latency-critical, but a regression here would make
`make ingest` painfully slow on a large knowledge base.
"""

from meeting_copilot.knowledge.chunking import chunk_text

_LARGE_TEXT = " ".join(f"token{i}" for i in range(50_000))


def test_chunk_text_throughput(benchmark):
    result = benchmark(chunk_text, _LARGE_TEXT, 500, 75)
    assert len(result) > 1
