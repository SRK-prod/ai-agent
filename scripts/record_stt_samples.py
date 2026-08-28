#!/usr/bin/env python3
"""Record a real audio sample for the STT model A/B comparison.

    python scripts/record_stt_samples.py --duration 20 --label short-technical-q
    python scripts/record_stt_samples.py --duration 45 --label long-architecture-q
    python scripts/record_stt_samples.py --checklist   # show coverage without recording

Record from the SAME source the live pipeline hears -- play back a real interview
recording, or have someone ask you real questions through the call app -- so the
comparison reflects actual call-audio quality, not a clean microphone reading.

A single easy sample isn't a real comparison. Cover, ideally one each:
  short-technical-q, medium-technical-q, long-architecture-q, cloud-terminology,
  numbers-acronyms, follow-up-q, natural-pauses, noisy-call-audio
(run with --checklist to see this with your current coverage checked off -- this is
exactly the noisy/technical-terminology combination that caused the 2026-08-07 regression
this comparison exists to avoid repeating blind).

Saved under data/stt_samples/ (gitignored, local only). Run
scripts/compare_stt_models.py once you have a few, and delete the directory
(`--clean` on that script) when done -- these are real interview recordings.
"""

import argparse

from meeting_copilot.stt.ab_harness import print_checklist, record_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds to record")
    parser.add_argument("--label", default="sample", help="Short name for this sample")
    parser.add_argument(
        "--checklist", action="store_true", help="Show coverage checklist and exit"
    )
    args = parser.parse_args()

    if args.checklist:
        print_checklist()
        return
    record_sample(args.duration, args.label)


if __name__ == "__main__":
    main()
