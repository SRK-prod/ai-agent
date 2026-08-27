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
    "Total time from a question being ready to process to its answer being fully "
    "generated and optimized -- NOT the perceived latency; see TTFA_SECONDS for that",
    buckets=(0.1, 0.25, 0.5, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6),
)

# The metric that actually decides whether this is usable live: from a speech segment
# being ready to process (diarize+STT not yet run) to the first LLM token reaching the
# overlay. Everything before this point is dead air the candidate is standing in. Separate
# from PIPELINE_TOTAL_LATENCY_SECONDS above, which measures time to the COMPLETE answer --
# a candidate doesn't need to wait for that to start reading.
TTFA_SECONDS = Histogram(
    "meeting_copilot_ttfa_seconds",
    "Time-to-first-answer: from speech segment ready to first LLM token shown",
    buckets=(0.25, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4, 3.2, 4.8, 6.4, 9.6),
)

# Isolates the LLM API's own latency (call start -> first streamed token) from everything
# that happens before it (STT, context/prompt assembly, retrieval) -- lets a slow answer be
# attributed to "the setup before Claude" vs. "Claude itself" instead of guessing.
LLM_TTFT_SECONDS = Histogram(
    "meeting_copilot_llm_ttft_seconds",
    "Time from the LLM call starting to its first streamed token",
    buckets=(0.1, 0.25, 0.5, 0.8, 1.2, 1.6, 2.4, 3.2, 6.4),
)


class StageTimer:
    """`with StageTimer("stt") as t: ...` records into STAGE_LATENCY_SECONDS{stage="stt"}
    and leaves the measured duration on `t.elapsed_seconds` for callers that need to fold it
    into a per-question breakdown (see MeetingPipeline._process_transcript)."""

    def __init__(self, stage: str):
        self._stage = stage
        self._start: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> Self:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_seconds = time.monotonic() - self._start
        STAGE_LATENCY_SECONDS.labels(stage=self._stage).observe(self.elapsed_seconds)
