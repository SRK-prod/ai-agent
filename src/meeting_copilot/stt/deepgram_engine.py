"""Deepgram transcription engine -- a drop-in replacement for the local Whisper engines.

Why this exists: on a CPU-only machine STT is the entire latency budget AND it competes
with the video call for the same cores. Measured on a 2-core i5-7300U, 2026-09-01:

    same ~2.2s utterance, faster-whisper "base"
        CPU idle          3.85s
        CPU at 99% (Chrome + Docker + the call running)   33.2s

There is no local model setting that is both fast and accurate under that contention --
bigger models fix the mis-transcriptions and blow the latency budget, smaller ones do the
reverse. Sending the audio out instead removes STT from the CPU entirely: the cost becomes
a network round trip (~0.3-0.6s for a short utterance) that does not degrade when the
machine is busy, and Deepgram's accuracy on conversational call audio is well above the
small local models.

TRADE-OFF, stated plainly: this sends meeting audio to a third party. That is a real change
from the local-first design -- the whole point of local Whisper was that raw interview audio
never left the machine. The question text and persona already go to Anthropic, so the trust
boundary is not new, but the raw AUDIO leaving is. Keep stt.backend on faster-whisper for
anything that must stay on-device.

Implements the same interface as FasterWhisperEngine/MlxWhisperEngine --
`transcribe_samples(samples, sample_rate) -> str` -- so SttStage, the hallucination guard,
the term normalizer and everything downstream are unchanged.
"""

from __future__ import annotations

import io
import wave

import httpx
import numpy as np

from meeting_copilot.config import SttConfig, get_config
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

_ENDPOINT = "https://api.deepgram.com/v1/listen"


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1,1] -> 16-bit PCM WAV in memory (no temp file, no ffmpeg hop)."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _encode(samples: np.ndarray, sample_rate: int) -> tuple[bytes, str]:
    """Encode an utterance for upload, returning (body, content_type).

    FLAC by preference: it is LOSSLESS, so transcription accuracy is identical to WAV, but
    roughly half the bytes. That matters because the upload shares the link with the video
    call itself -- measured 2026-09-01, a 48s WAV (1.5MB) hit the write timeout and the
    utterance was lost, while smaller bodies went through in well under a second. Halving
    the payload halves the exposure to that.

    Falls back to WAV if libsndfile has no FLAC support in this build, since a slightly
    larger request is obviously better than no transcription.
    """
    try:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, np.clip(samples, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_16")
        return buf.getvalue(), "audio/flac"
    except Exception:
        logger.debug("FLAC encode unavailable, sending WAV", exc_info=True)
        return _to_wav_bytes(samples, sample_rate), "audio/wav"


class DeepgramEngine:
    def __init__(self, config: SttConfig | None = None):
        self._cfg = config or get_config().stt
        self._api_key = get_config().secrets.require_deepgram_key()
        # One client, reused, with the idle connection held open for the whole interview.
        # Measured from this machine 2026-09-01 (78KB body, a 2.5s utterance):
        #     cold, including DNS + TLS handshake   1044 ms
        #     warm, connection reused                247 ms
        # httpx's default keepalive_expiry is only 5s, and interview questions are minutes
        # apart -- so with the default EVERY question would pay the cold path and this engine
        # would hand back most of the latency it exists to save. Holding one idle connection
        # open costs nothing and keeps every question on the 247ms path.
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._cfg.cloud_timeout_seconds),
            limits=httpx.Limits(
                max_keepalive_connections=1,
                keepalive_expiry=self._cfg.cloud_keepalive_seconds,
            ),
            headers={"Authorization": f"Token {self._api_key}"},
        )
        logger.info(f"Deepgram STT engine ready (model={self._cfg.deepgram_model})")

    def _params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "model": self._cfg.deepgram_model,
            "language": self._cfg.language,
            "punctuate": "true",
            "smart_format": "true",
            # No diarization/utterance splitting: Silero VAD already gave us exactly one
            # utterance, and asking Deepgram to re-segment it only adds work and latency.
            "diarize": "false",
        }
        # Deepgram's keyterm prompting is the direct equivalent of Whisper's initial_prompt,
        # so the domain vocabulary that stops "OOMKilled" becoming "OOM killed" carries over
        # rather than being re-solved. nova-3 uses `keyterm`; older models use `keywords`.
        # Was reusing initial_prompt -- the 28-term hint tuned for Whisper's decoder.
        # Deepgram takes up to 100 boost terms and does not suffer the attention
        # dilution that kept that list short, so it gets its own fuller list.
        terms = self._cfg.keyterms
        if terms:
            key = "keyterm" if self._cfg.deepgram_model.startswith("nova-3") else "keywords"
            params[key] = terms[: self._cfg.deepgram_max_keyterms]
        return params

    def transcribe_samples(self, samples, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(
                f"DeepgramEngine expects 16kHz audio, got {sample_rate}Hz -- "
                "resample before calling transcribe_samples."
            )
        body, content_type = _encode(samples, sample_rate)

        # One retry. The observed failure mode is a transient write timeout when the upload
        # competes with the video call for bandwidth, and a retry on a fresh connection
        # costs far less than silently dropping a real interview question. Bounded at two
        # attempts so a genuine outage still fails fast rather than eating the answer window.
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = self._client.post(
                    _ENDPOINT,
                    params=self._params(),
                    content=body,
                    headers={"Content-Type": content_type},
                )
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:  # noqa: BLE001 -- see below; must never reach the pipeline
                last_error = exc
                if attempt == 1:
                    logger.warning(
                        f"Deepgram request failed ({type(exc).__name__}), retrying once "
                        f"[{len(body) / 1024:.0f}KB {content_type}]"
                    )
        else:
            # Never raise into the pipeline: a transcription failure must cost one question,
            # not the whole session. SttStage treats "" as "nothing was said" and moves on.
            logger.error(
                f"Deepgram failed twice, dropping this utterance: {type(last_error).__name__}: "
                f"{last_error}"
            )
            return ""

        try:
            alt = payload["results"]["channels"][0]["alternatives"][0]
        except (KeyError, IndexError):
            logger.warning(f"Unexpected Deepgram response shape: {str(payload)[:200]}")
            return ""
        return str(alt.get("transcript", "")).strip()
