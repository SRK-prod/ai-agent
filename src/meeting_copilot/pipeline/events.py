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
    """Point-in-time read of whether the audio CAPTURE INFRASTRUCTURE is actually working --
    see audio/capture.py AudioCapture.health(). Answers "can this still hear the
    interviewer if they speak", not "has the interviewer spoken recently" -- this capture
    stream carries only the interviewer's side (BlackHole/system audio), so it is silent for
    the entire time the candidate is answering (routinely 30s-several minutes), which is
    normal and must never by itself read as a failure. Confirmed live 2026-08-25: an
    earlier version that escalated on silence DURATION produced 7 false AUDIO_INPUT_LOST
    warnings in one ~11-minute session, every one recovering the instant the interviewer
    spoke again.
    """

    state: str  # "AUDIO_ACTIVE" | "AUDIO_SILENT" | "AUDIO_INPUT_LOST"
    # AUDIO_ACTIVE: None. AUDIO_SILENT: "NO_INTERVIEWER_SPEECH" -- informational, not an
    # error; can legitimately last minutes. AUDIO_INPUT_LOST: "CALLBACK_STALLED" (the
    # capture callback itself stopped firing) | "DEVICE_UNAVAILABLE" (the configured input
    # device is gone) | "DEVICE_CHANGED" (that device index now belongs to a different
    # device -- CoreAudio renumbering, e.g. after a Bluetooth device connects). Not shown in
    # the overlay, logged on every state transition -- this is exactly the detail that was
    # missing to diagnose past live dropouts after the fact.
    reason: str | None
    seconds_since_signal: float
    seconds_since_callback: float
    last_peak: float
    last_rms: float
    callback_count: int
