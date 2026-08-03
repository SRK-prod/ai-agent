"""Faster-Whisper transcription of DiarizedSegments that are NOT the enrolled user.

Faster-Whisper doesn't do true token-level streaming, so "streaming" here
means: VAD already gave us short (~a few second) utterance-level segments,
and each one is transcribed as a single blocking call run in a thread pool
so it doesn't stall the asyncio event loop. See configs/settings.yaml
`stt.model_size` to trade latency for accuracy on your hardware.
"""

from __future__ import annotations

import asyncio

import torch
from faster_whisper import WhisperModel

from meeting_copilot.cache.redis_cache import RedisCache
from meeting_copilot.config import SttConfig, get_config
from meeting_copilot.pipeline.events import DiarizedSegment, Transcript
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # faster-whisper (CTranslate2) doesn't support MPS; falls back to CPU


class FasterWhisperEngine:
    def __init__(self, config: SttConfig | None = None):
        self._cfg = config or get_config().stt
        device = _resolve_device(self._cfg.device)
        logger.info(
            f"Loading Faster-Whisper model={self._cfg.model_size} "
            f"device={device} compute_type={self._cfg.compute_type}"
        )
        self._model = WhisperModel(
            self._cfg.model_size, device=device, compute_type=self._cfg.compute_type
        )

    def transcribe_samples(self, samples, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(
                f"FasterWhisperEngine expects 16kHz audio, got {sample_rate}Hz -- "
                "resample before calling transcribe_samples."
            )
        segments, _info = self._model.transcribe(
            samples,
            language=self._cfg.language,
            condition_on_previous_text=True,
            vad_filter=False,  # our own Silero VAD already gated this to a speech segment
        )
        return " ".join(s.text.strip() for s in segments).strip()


def get_stt_engine(config: SttConfig | None = None):
    """Engine factory driven by `stt.backend` (faster-whisper = CPU, mlx-whisper = M-series GPU)."""
    cfg = config or get_config().stt
    if cfg.backend == "mlx-whisper":
        from meeting_copilot.stt.mlx_whisper_engine import MlxWhisperEngine

        return MlxWhisperEngine(cfg)
    return FasterWhisperEngine(cfg)


class SttStage:
    """Pipeline-facing wrapper: DiarizedSegment -> Transcript | None."""

    def __init__(self, engine=None, cache: RedisCache | None = None):
        self._engine = engine or get_stt_engine()
        self._cache = cache or RedisCache()

    async def transcribe(self, diarized: DiarizedSegment) -> Transcript | None:
        if diarized.is_me:
            return None

        text = await asyncio.to_thread(
            self._engine.transcribe_samples,
            diarized.segment.samples,
            diarized.segment.sample_rate,
        )
        if not text:
            return None

        if await self._cache.seen_recently(diarized.speaker_id, text):
            logger.debug(f"Skipping duplicate utterance from {diarized.speaker_id}: {text!r}")
            return None

        return Transcript(
            speaker_id=diarized.speaker_id,
            text=text,
            start_time=diarized.segment.start_time,
            end_time=diarized.segment.end_time,
            language=self._engine._cfg.language,
        )
