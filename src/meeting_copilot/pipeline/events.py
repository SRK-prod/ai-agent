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
    # True if the text grammatically reads as a question (a "?" or an interrogative
    # opener like "how"/"what would you"). False for a detection that survived ONLY on a
    # keyword match or technical-term density -- e.g. "production BigPanda application
    # suddenly produces false alarms" matches the keyword "bigpanda" and is a valid,
    # answerable sentence, but it is declarative, not an ask. The orchestrator uses this to
    # avoid letting a keyword-dense scenario-setup sentence cut off an in-progress
    # multi-sentence scenario buildup and get answered prematurely.
    has_interrogative_signal: bool = True


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


@dataclass
class AudioHealth:
    """Point-in-time read of whether the audio capture stream is actually delivering usable
    signal -- see audio/capture.py AudioCapture.health(). Distinguishes "the interviewer
    is quiet" from "the pipeline stopped hearing them", which look identical downstream
    (no transcripts) but need very different responses.
    """

    state: str  # "AUDIO_ACTIVE" | "AUDIO_SILENT" | "AUDIO_INPUT_LOST"
    # Only set when state != AUDIO_ACTIVE. "CALLBACK_STALLED" (the capture callback itself
    # stopped firing -- device/stream problem) vs "ZERO_SIGNAL" (callbacks are arriving but
    # every buffer is silent -- could be a routing problem, e.g. the Multi-Output/Aggregate
    # device losing the actual source). Not shown in the overlay, logged on every state
    # transition -- this is exactly the detail that was missing to diagnose past live
    # dropouts after the fact.
    reason: str | None
    seconds_since_signal: float
    seconds_since_callback: float
    last_peak: float
    last_rms: float
    callback_count: int
