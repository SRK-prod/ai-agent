"""Silero VAD: gates silence out of the stream and assembles SpeechSegments.

Uses the `silero-vad` PyPI package's VADIterator, which requires fixed-size
windows (512 samples at 16kHz, 256 at 8kHz) -- input AudioFrames are
re-chunked into a rolling buffer to match, since our capture block size
(configs/settings.yaml audio.block_ms) won't line up with that exactly.

Model weights auto-download on first use (via torch.hub cache) -- no token
required, unlike pyannote.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from meeting_copilot.audio.preprocess import preprocess_frame
from meeting_copilot.config import VadConfig, get_config
from meeting_copilot.pipeline.events import AudioFrame, SpeechSegment
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


class SileroVAD:
    def __init__(self, config: VadConfig | None = None, sample_rate: int | None = None):
        app_cfg = get_config()
        self._cfg = config or app_cfg.vad
        self._sample_rate = sample_rate or app_cfg.audio.sample_rate
        self._window_samples = 512 if self._sample_rate == 16000 else 256

        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=self._cfg.threshold,
            sampling_rate=self._sample_rate,
            min_silence_duration_ms=self._cfg.min_silence_ms,
        )

        self._buffer = np.zeros(0, dtype=np.float32)
        self._speech_chunks: list[np.ndarray] = []
        self._speech_start_time: float | None = None
        self._in_speech = False

    def _reset_speech_state(self) -> None:
        self._speech_chunks = []
        self._speech_start_time = None
        self._in_speech = False
        self._iterator.reset_states()

    async def segments(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[SpeechSegment]:
        async for raw_frame in frames:
            frame = preprocess_frame(raw_frame)
            self._buffer = np.concatenate([self._buffer, frame.samples])

            while len(self._buffer) >= self._window_samples:
                chunk = self._buffer[: self._window_samples]
                self._buffer = self._buffer[self._window_samples :]

                if self._in_speech:
                    self._speech_chunks.append(chunk)

                event = self._iterator(chunk, return_seconds=False)
                if not event:
                    continue

                if "start" in event and not self._in_speech:
                    self._in_speech = True
                    self._speech_start_time = frame.timestamp
                    self._speech_chunks = [chunk]
                elif "end" in event and self._in_speech:
                    samples = np.concatenate(self._speech_chunks) if self._speech_chunks else chunk
                    duration_ms = len(samples) / self._sample_rate * 1000
                    start_time = self._speech_start_time or frame.timestamp
                    self._reset_speech_state()
                    if duration_ms >= self._cfg.min_speech_ms:
                        logger.debug(f"VAD speech segment: {duration_ms:.0f}ms")
                        yield SpeechSegment(
                            samples=samples,
                            sample_rate=self._sample_rate,
                            start_time=start_time,
                            end_time=frame.timestamp,
                        )
