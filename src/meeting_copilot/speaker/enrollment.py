"""One-time voice enrollment: record 30-60s, embed, store.

Driven by scripts/enroll_voice.py (`make enroll`). This is a blocking,
offline CLI flow -- not part of the live async pipeline.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from meeting_copilot.audio.capture import resolve_device
from meeting_copilot.config import get_config
from meeting_copilot.pipeline.events import SpeechSegment
from meeting_copilot.speaker.diarization import SpeakerEmbedder
from meeting_copilot.speaker.identity import EnrollmentStore
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

MIN_DURATION_SECONDS = 30.0
MAX_DURATION_SECONDS = 60.0


def record_voice(duration_seconds: float, sample_rate: int, device: int | None = None) -> np.ndarray:
    logger.info(f"Recording {duration_seconds:.0f}s for voice enrollment -- please talk naturally.")
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio[:, 0]


def enroll_from_samples(samples: np.ndarray, sample_rate: int) -> None:
    duration = len(samples) / sample_rate
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            f"Enrollment recording is only {duration:.1f}s; need at least "
            f"{MIN_DURATION_SECONDS:.0f}s for a reliable voice embedding."
        )
    segment = SpeechSegment(
        samples=samples, sample_rate=sample_rate, start_time=0.0, end_time=duration
    )
    embedder = SpeakerEmbedder()
    embedding = embedder.embed(segment)
    EnrollmentStore().save(embedding, get_config().speaker.embedding_model)
    logger.info("Voice enrollment complete.")


def enroll_interactive(duration_seconds: float = 45.0) -> None:
    duration_seconds = min(max(duration_seconds, MIN_DURATION_SECONDS), MAX_DURATION_SECONDS)
    cfg = get_config()
    device = resolve_device(cfg.audio.input_device)
    samples = record_voice(duration_seconds, cfg.audio.sample_rate, device=device)
    enroll_from_samples(samples, cfg.audio.sample_rate)
