#!/usr/bin/env python3
"""Pre-download every model the pipeline needs, so the first real meeting
isn't slowed down by cold downloads. Run once after `make setup` -- see
`make download-models`.
"""

from __future__ import annotations

from meeting_copilot.config import get_config
from meeting_copilot.utils.logging import configure_logging, get_logger

logger = get_logger()


def main() -> None:
    configure_logging()
    cfg = get_config()

    logger.info("Downloading Silero VAD (no credentials required)...")
    from silero_vad import load_silero_vad

    load_silero_vad()

    logger.info(f"Downloading Faster-Whisper model '{cfg.stt.model_size}'...")
    from meeting_copilot.stt.faster_whisper_engine import FasterWhisperEngine

    FasterWhisperEngine(cfg.stt)

    logger.info(f"Downloading local embedding model '{cfg.knowledge.embedding_model}'...")
    from meeting_copilot.knowledge.embeddings import LocalEmbedder

    LocalEmbedder(cfg.knowledge)

    if cfg.secrets.hf_token:
        logger.info(f"Downloading pyannote embedding model '{cfg.speaker.embedding_model}'...")
        from meeting_copilot.speaker.diarization import SpeakerEmbedder

        SpeakerEmbedder(cfg.speaker)
    else:
        logger.warning(
            "HF_TOKEN not set -- skipping pyannote model download. "
            "Set it in .env and re-run before enrolling/running a real meeting "
            "(see docs/installation.md)."
        )

    logger.info("Model download complete.")


if __name__ == "__main__":
    main()
