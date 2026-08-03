#!/usr/bin/env python3
"""Thin wrapper: `python scripts/enroll_voice.py [--duration 45]` -- see `make enroll`."""

from meeting_copilot.scripts_entry import enroll_voice_main

if __name__ == "__main__":
    enroll_voice_main()
