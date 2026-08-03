#!/usr/bin/env python3
"""Thin wrapper: `python scripts/ingest_knowledge.py --topics configs/topics.yaml` -- see `make ingest`."""

from meeting_copilot.scripts_entry import ingest_knowledge_main

if __name__ == "__main__":
    ingest_knowledge_main()
