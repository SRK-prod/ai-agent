"""Console-script entrypoints (see pyproject.toml [project.scripts]).

The scripts/ directory files are thin wrappers around these same functions,
for people running from a source checkout without `pip install -e .`.
"""

from __future__ import annotations


def enroll_voice_main() -> None:
    import argparse

    from meeting_copilot.speaker.enrollment import MAX_DURATION_SECONDS, enroll_interactive

    parser = argparse.ArgumentParser(description="Enroll your voice (30-60s recording)")
    parser.add_argument("--duration", type=float, default=45.0, help="Recording length in seconds")
    args = parser.parse_args()
    enroll_interactive(min(args.duration, MAX_DURATION_SECONDS))


def ingest_knowledge_main() -> None:
    from meeting_copilot.knowledge.ingestion import _cli

    _cli()
