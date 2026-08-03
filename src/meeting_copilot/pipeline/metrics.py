"""Prometheus histograms, one per pipeline stage, exposed at /metrics by server/api.py."""

from __future__ import annotations

import time
from typing import Self

from prometheus_client import Histogram

STAGE_LATENCY_SECONDS = Histogram(
    "meeting_copilot_stage_latency_seconds",
    "Latency of a single pipeline stage",
    labelnames=["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8),
)

PIPELINE_TOTAL_LATENCY_SECONDS = Histogram(
    "meeting_copilot_total_latency_seconds",
    "End-to-end latency from speech segment end to answer emitted",
    buckets=(0.1, 0.25, 0.5, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6),
)

class StageTimer:
    """`with StageTimer("stt"): ...` records into STAGE_LATENCY_SECONDS{stage="stt"}."""

    def __init__(self, stage: str):
        self._stage = stage
        self._start: float = 0.0

    def __enter__(self) -> Self:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        STAGE_LATENCY_SECONDS.labels(stage=self._stage).observe(time.monotonic() - self._start)
