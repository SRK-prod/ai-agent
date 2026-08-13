"""Apple-Silicon-GPU Whisper transcription via mlx-whisper (Metal).

faster-whisper's CTranslate2 engine is CPU-only on macOS; mlx-whisper runs
the same Whisper models on the M-series GPU instead. Measured on this
project's dev machine (warm): 6s audio = 0.33s, 30s audio = 0.32s -- vs
1.1s / ~5.5s for the CPU `small` model, i.e. transcription time becomes
effectively flat regardless of how long the speaker talked.

Only available on Apple Silicon (pyproject marks the dependency
darwin/arm64-only); `stt.backend` in configs/settings.yaml picks between
this and FasterWhisperEngine.
"""

from __future__ import annotations

import numpy as np

from meeting_copilot.config import SttConfig, get_config
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

# Map the generic stt.model_size names onto mlx-community's converted repos,
# so switching backends doesn't require changing model_size.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


class MlxWhisperEngine:
    def __init__(self, config: SttConfig | None = None):
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        self._cfg = config or get_config().stt
        self._repo = _MLX_REPOS.get(self._cfg.model_size)
        if self._repo is None:
            raise ValueError(
                f"stt.model_size {self._cfg.model_size!r} has no known mlx-community repo; "
                f"choose one of {sorted(_MLX_REPOS)} or switch stt.backend to faster-whisper."
            )
        logger.info(f"Loading mlx-whisper model {self._repo} (Apple Silicon GPU)")
        # Warm up: first call loads weights (and downloads them on very first use).
        self._mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32), path_or_hf_repo=self._repo, language=self._cfg.language
        )

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(
                f"MlxWhisperEngine expects 16kHz audio, got {sample_rate}Hz -- "
                "resample before calling transcribe_samples."
            )
        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self._repo,
            language=self._cfg.language,
            initial_prompt=self._cfg.vocabulary_hint,
            # condition_on_previous_text=False is the single most important accuracy setting
            # here. With it True (the default), each window is conditioned on the previously
            # decoded text -- so one garbled decode cascades, producing the runaway repetition
            # loops seen live ("Disability Disability Disability...", "create create create...").
            # Disabling it costs a little cross-window coherence but stops the cascade entirely.
            condition_on_previous_text=False,
            # Greedy decode: deterministic, and avoids the temperature-fallback ladder
            # (0.0->1.0) that produces increasingly invented text on hard audio.
            temperature=0.0,
            # Reject a window whose gzip compression ratio implies heavy repetition. Default
            # 2.4 is lenient; 2.0 catches repetition loops earlier.
            compression_ratio_threshold=2.0,
            # Explicitly drop segments Whisper decodes over near-silence -- the other main
            # hallucination source ("Thank you." over an empty channel).
            hallucination_silence_threshold=2.0,
            no_speech_threshold=0.6,
        )
        text = result.get("text", "")
        return text.strip() if isinstance(text, str) else ""
