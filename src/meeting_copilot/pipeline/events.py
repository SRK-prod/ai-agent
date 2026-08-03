"""Typed events passed between pipeline stages (audio -> ... -> answer).

Each stage in meeting_copilot.pipeline.orchestrator consumes one event type
and produces the next, so the contract between stages is explicit and each
stage is independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class AudioFrame:
    """A single raw capture block, mono float32 PCM in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int
    timestamp: float  # seconds since pipeline start


@dataclass(eq=False)
class SpeechSegment:
    """A contiguous span of speech, assembled by VAD from one or more AudioFrames."""

    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float


@dataclass(eq=False)
class DiarizedSegment:
    """A SpeechSegment attributed to a speaker."""

    segment: SpeechSegment
    speaker_id: str  # "me" | "speaker_a" | "speaker_b" | ...
    is_me: bool
    similarity_to_me: float


@dataclass
class Transcript:
    """Text transcribed for a DiarizedSegment that was not the enrolled user."""

    speaker_id: str
    text: str
    start_time: float
    end_time: float
    language: str = "en"


@dataclass
class DetectedQuestion:
    """A Transcript flagged as worth answering, with the reason it matched."""

    transcript: Transcript
    matched_keywords: list[str] = field(default_factory=list)
    ends_with_question_mark: bool = False


@dataclass
class RetrievedChunk:
    text: str
    source: str
    topic: str
    score: float


@dataclass
class RetrievedContext:
    question: DetectedQuestion
    chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class Answer:
    question: DetectedQuestion
    text: str
    format_type: str  # "prose" | "bullets" | "code" | "table"
    confidence: float
    low_confidence: bool
