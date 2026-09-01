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

        # 0 disables the cap (the original behaviour: only silence ever ends a segment).
        self._max_speech_samples = int(self._cfg.max_speech_ms / 1000 * self._sample_rate)

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

                    # FORCED CUT ON A LONG UTTERANCE. Without this the only thing that ever
                    # closes a segment is min_silence_ms of silence, so an interviewer who
                    # talks continuously produces one enormous segment and NOTHING reaches
                    # STT until they finally stop. Measured live 2026-09-01: a 28.9s segment
                    # took 11.1s to decode, so the answer appeared 14s after the question --
                    # while a normal 2.4s segment answered in 5.1s.
                    #
                    # CUT AT THE WINDOW BOUNDARY, NOT SOONER. Whisper decodes in fixed
                    # 30-second windows, so cost is flat inside a window and steps sharply
                    # at the boundary -- measured (base, CPU): 12s->2.98s, 24s->3.56s,
                    # 28s->4.54s, then 32s->7.66s once a second window is needed. Cutting
                    # EARLIER than the boundary therefore makes things worse, not better: it
                    # turns one window's work into two. max_speech_ms is sized just under 30s
                    # for that reason, and lowering it to "get answers sooner" is a trap.
                    #
                    # A cut mid-question is handled downstream: the orchestrator's fragment
                    # merge (_FRAGMENT_MERGE_GAP_SECONDS) and answer-revision window fold the
                    # continuation into the same answer rather than starting a second one.
                    if (
                        self._max_speech_samples
                        and sum(len(c) for c in self._speech_chunks) >= self._max_speech_samples
                    ):
                        samples = np.concatenate(self._speech_chunks)
                        start_time = self._speech_start_time or frame.timestamp
                        duration_ms = len(samples) / self._sample_rate * 1000
                        # Keep _in_speech True and carry on accumulating -- this is a cut in a
                        # continuing utterance, not the end of one, so the VAD iterator's own
                        # state is deliberately NOT reset.
                        self._speech_chunks = []
                        self._speech_start_time = frame.timestamp
                        logger.debug(f"VAD forced cut at max length: {duration_ms:.0f}ms")
                        yield SpeechSegment(
                            samples=samples,
                            sample_rate=self._sample_rate,
                            start_time=start_time,
                            end_time=frame.timestamp,
                        )

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
