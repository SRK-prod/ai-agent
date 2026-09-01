"""Faster-Whisper transcription of DiarizedSegments that are NOT the enrolled user.

Faster-Whisper doesn't do true token-level streaming, so "streaming" here
means: VAD already gave us short (~a few second) utterance-level segments,
and each one is transcribed as a single blocking call run in a thread pool
so it doesn't stall the asyncio event loop. See configs/settings.yaml
`stt.model_size` to trade latency for accuracy on your hardware.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import torch
from faster_whisper import WhisperModel

from meeting_copilot.cache.redis_cache import RedisCache
from meeting_copilot.config import SttConfig, get_config
from meeting_copilot.pipeline.events import DiarizedSegment, Transcript
from meeting_copilot.stt.term_normalizer import normalize as normalize_terms
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
            self._cfg.model_size,
            device=device,
            compute_type=self._cfg.compute_type,
            cpu_threads=self._cfg.cpu_threads,
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
            beam_size=self._cfg.beam_size,
            # False, not True: each VAD segment is an independent utterance here, so there is
            # no previous text worth conditioning on -- and conditioning is the known trigger
            # for Whisper's repetition-loop hallucinations (the exact failure _is_hallucinated
            # below exists to catch). Measured no transcript regression, slightly faster.
            condition_on_previous_text=False,
            vad_filter=False,  # our own Silero VAD already gated this to a speech segment
            # Skip timestamp-token generation: the pipeline only ever uses the joined text
            # (segment boundaries come from Silero VAD, not from Whisper), so the decoder is
            # paying to emit tokens nothing reads. Measured 2.13s -> 1.76s on this CPU.
            without_timestamps=True,
            initial_prompt=self._cfg.initial_prompt,
        )
        return " ".join(s.text.strip() for s in segments).strip()


# Measured 2026-08-26 on this project's dev machine: pure digital silence (all-zero
# samples) fed to large-v3-turbo still hallucinates text ('and the other one.') and pays
# the model's full ~2.3-2.7s encoder floor to produce it -- Whisper always encodes a fixed
# 30s window internally regardless of how little real signal is in it. RMS this low is
# below any audible real speech (quiet real speech measured well above 0.01), so skipping
# the STT call entirely here is safe: it only short-circuits segments that were never
# going to produce real content, saving the wasted decode time and preventing the
# hallucinated text from ever reaching the question detector.
_SILENCE_RMS_THRESHOLD = 0.003


def _is_near_silent(samples: np.ndarray) -> bool:
    if samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return rms < _SILENCE_RMS_THRESHOLD


def _is_hallucinated(text: str) -> bool:
    """Whisper's most common failure mode on noisy/cross-talk/low-signal audio isn't
    silence -- it's a repetition loop, the same word or short phrase output dozens of
    times ('Disability Disability Disability...', 'diss diss diss...'). Real speech
    essentially never repeats one word this often, so a high repetition ratio is a
    reliable signal to discard the segment rather than treat it as real content."""
    words = [w.lower().strip(".,!?") for w in text.split()]
    if len(words) < 6:
        return False

    # Single-word repetition loop ("create create create...").
    counts = Counter(words)
    _most_common_word, most_common_count = counts.most_common(1)[0]
    if most_common_count >= 5 and most_common_count / len(words) > 0.35:
        return True

    # Short-PHRASE repetition loop ("the other one, the other one, ..." / "I could be
    # red, or I could be red, or ..."). Added 2026-08-26 after live review: a 2-5 word
    # repeating phrase dilutes any single word's frequency below the check above even
    # though the same phrase dominates the transcript -- e.g. a 3-word phrase repeated
    # puts each word at ~33% of total words, just under the 35% cutoff. Check n-grams of
    # length 2-5 for the same "one repeated unit covers most of the text" signal.
    for n in (2, 3, 4, 5):
        if len(words) < n * 3:  # need at least 3 repeats to call it a loop
            continue
        ngram_counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        _phrase, phrase_count = ngram_counts.most_common(1)[0]
        if phrase_count >= 4 and (phrase_count * n) / len(words) > 0.5:
            return True

    return False


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

        if _is_near_silent(diarized.segment.samples):
            # Skip the expensive model call entirely -- near-silent audio (VAD passed it
            # through, but there's effectively no signal) still costs Whisper's full
            # ~2.3-2.7s encoder floor AND reliably hallucinates fake text from it, which
            # was cascading into false question-revision triggers downstream.
            logger.debug("Skipping STT on near-silent segment (below RMS threshold)")
            return None

        text = await asyncio.to_thread(
            self._engine.transcribe_samples,
            diarized.segment.samples,
            diarized.segment.sample_rate,
        )
        if not text:
            return None

        if _is_hallucinated(text):
            # Full text, not truncated -- a discard is a real interview question silently
            # dropped if this is ever wrong, so this needs to be verifiable from the log
            # alone rather than requiring the reader to re-run the classifier by hand.
            logger.warning(f"Discarding likely STT hallucination: {text!r}")
            return None

        # Repair mis-transcribed technical vocabulary BEFORE question detection. Whisper
        # mangles domain terms ("OOMCade" for OOMKilled), and the detector scores on keyword
        # matches -- so without this a real technical question gets scored as noise and
        # silently dropped. Deterministic, ~0.5ms, no model call.
        normalized, repairs = normalize_terms(text)
        if repairs:
            logger.info(f"STT term recovery ({repairs}): {text[:60]!r} -> {normalized[:60]!r}")
            text = normalized

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
