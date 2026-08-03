"""Noise removal and normalization.

Two tiers, because spectral noise reduction is too expensive/ineffective to
run on every ~30ms capture block:

- `preprocess_frame`: cheap, real-time-safe DC-offset removal applied to every
  AudioFrame before it reaches VAD.
- `preprocess_segment`: spectral noise reduction + peak normalization applied
  once per assembled SpeechSegment (a full utterance), before it goes to STT.
"""

from __future__ import annotations

import noisereduce as nr
import numpy as np

from meeting_copilot.config import AudioConfig, get_config
from meeting_copilot.pipeline.events import AudioFrame, SpeechSegment


def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    return samples - np.mean(samples)


def normalize_peak(samples: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak < 1e-8:
        return samples
    return samples * (target_peak / peak)


def reduce_noise(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return nr.reduce_noise(y=samples, sr=sample_rate, stationary=True)


def preprocess_frame(frame: AudioFrame, config: AudioConfig | None = None) -> AudioFrame:
    cfg = config or get_config().audio
    samples = remove_dc_offset(frame.samples)
    if not cfg.noise_reduction:
        return AudioFrame(samples=samples, sample_rate=frame.sample_rate, timestamp=frame.timestamp)
    return AudioFrame(samples=samples, sample_rate=frame.sample_rate, timestamp=frame.timestamp)


def preprocess_segment(
    segment: SpeechSegment, config: AudioConfig | None = None
) -> SpeechSegment:
    cfg = config or get_config().audio
    samples = segment.samples
    if cfg.noise_reduction:
        samples = reduce_noise(samples, segment.sample_rate)
    samples = normalize_peak(samples)
    return SpeechSegment(
        samples=samples,
        sample_rate=segment.sample_rate,
        start_time=segment.start_time,
        end_time=segment.end_time,
    )
