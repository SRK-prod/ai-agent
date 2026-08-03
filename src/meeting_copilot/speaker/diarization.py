"""Per-segment speaker embedding + assignment.

True continuous streaming diarization isn't native to pyannote.audio (its
pipeline is built for offline, whole-file processing). The pragmatic
approximation used here: each VAD-derived SpeechSegment (~a few seconds,
one dominant speaker in the common single-speaker-talking-at-a-time meeting
case) is embedded as a whole and compared against known speakers --
see speaker/identity.py for the assignment logic and configs/settings.yaml
`speaker.window_seconds` for tuning.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import torch
from pyannote.audio import Inference, Model

from meeting_copilot.config import SpeakerConfig, get_config
from meeting_copilot.pipeline.events import DiarizedSegment, SpeechSegment
from meeting_copilot.speaker.identity import SpeakerIdentity
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


@contextlib.contextmanager
def _trust_pyannote_checkpoint() -> Iterator[None]:
    """PyTorch 2.6+ defaults `torch.load` to `weights_only=True`, which rejects the
    pickled pytorch_lightning objects inside pyannote's official checkpoints
    (e.g. `pytorch_lightning.callbacks.early_stopping.EarlyStopping`). These are
    signed, official pyannote.audio models fetched from HuggingFace with our own
    token -- not arbitrary untrusted files -- so `weights_only=False` is the
    documented, safe mitigation here. Scoped to just this load, not global.
    """
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False  # force, not setdefault -- callers pass it explicitly
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


class SpeakerEmbedder:
    def __init__(self, config: SpeakerConfig | None = None):
        self._cfg = config or get_config().speaker
        hf_token = get_config().secrets.require_hf_token()
        with _trust_pyannote_checkpoint():
            model = Model.from_pretrained(self._cfg.embedding_model, use_auth_token=hf_token)
        self._inference = Inference(model, window="whole")

    def embed(self, segment: SpeechSegment) -> np.ndarray:
        waveform = torch.from_numpy(segment.samples).float().unsqueeze(0)  # (channels, samples)
        embedding = self._inference({"waveform": waveform, "sample_rate": segment.sample_rate})
        return np.asarray(embedding)


class SpeakerDiarizer:
    """Combines embedding + identity assignment into a single SpeechSegment -> DiarizedSegment step."""

    def __init__(
        self,
        embedder: SpeakerEmbedder | None = None,
        identity: SpeakerIdentity | None = None,
    ):
        self._embedder = embedder or SpeakerEmbedder()
        self._identity = identity or SpeakerIdentity()

    def diarize(self, segment: SpeechSegment) -> DiarizedSegment:
        embedding = self._embedder.embed(segment)
        speaker_id, is_me, similarity = self._identity.assign_speaker(embedding)
        if is_me:
            logger.debug(f"Ignoring segment ({similarity:.2f} similarity to enrolled voice)")
        return DiarizedSegment(
            segment=segment,
            speaker_id=speaker_id,
            is_me=is_me,
            similarity_to_me=similarity,
        )
