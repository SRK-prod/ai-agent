#!/usr/bin/env python3
"""Compare STT models on the samples recorded by scripts/record_stt_samples.py.

    python scripts/compare_stt_models.py
    python scripts/compare_stt_models.py --models large-v3-turbo small
    python scripts/compare_stt_models.py --stt-only   # skip the real LLM call, decode speed only
    python scripts/compare_stt_models.py --clean      # delete samples after reporting

By default this measures the metric that matters -- end-of-speech to first visible answer
token -- which means it makes one real Anthropic API call per sample per model. Pass
--stt-only for a quick, free, but incomplete first pass (decode speed alone; doesn't catch
a model that decodes fast but loses or mangles the question before the LLM ever sees it).

Prints every model's transcript for every sample side by side, flags hallucination-loop
detections, technical-term mis-transcriptions, and questions the rule-based detector
wouldn't have recognized at all, then the latency and cost tables. Read the transcripts
before the numbers -- a faster model that mangles a term or loses the question is not
actually a win for a live interview.
"""

import argparse
import asyncio
import shutil

from meeting_copilot.stt.ab_harness import CANDIDATE_MODELS, report, run_comparison, samples_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(CANDIDATE_MODELS))
    parser.add_argument(
        "--stt-only",
        action="store_true",
        help="Skip the real LLM call -- decode speed only, no cost, incomplete picture",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete data/stt_samples/ after reporting"
    )
    args = parser.parse_args()

    results = asyncio.run(run_comparison(tuple(args.models), stt_only=args.stt_only))
    report(results)

    if args.clean:
        shutil.rmtree(samples_dir(), ignore_errors=True)
        print(f"\nDeleted {samples_dir()}")


if __name__ == "__main__":
    main()
