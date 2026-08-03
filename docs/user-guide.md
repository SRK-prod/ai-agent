# User Guide

## Day to day

1. Before the meeting (once, or whenever your topics change): update
   `configs/topics.yaml` and run `make ingest`.
2. Start the meeting app, then `make run`. No further interaction is needed
   -- the overlay appears in the top-right corner and starts showing answers
   as other participants ask technical questions. Your own speech is
   detected and silently ignored.

## Reading the overlay

- **Header**: shows "meeting-copilot" normally, or "meeting-copilot — low
  confidence" when Claude's own self-assessed confidence (see
  `docs/performance.md`) was below `llm.low_confidence_threshold` (default
  80%). Treat low-confidence answers as a starting point, not a citation.
- **Body**: the answer, 30-100 words, formatted as bullets (architecture),
  a table (tradeoffs), a code block (coding questions), or plain prose.
- **Footer**: the confidence percentage.

## Hotkeys

Configurable in `configs/settings.yaml` under `overlay.hotkeys` (defaults
below):

| Action | Default | Effect |
|---|---|---|
| Hide | `Cmd+Shift+H` | Toggle the overlay's visibility |
| Pin | `Cmd+Shift+P` | Toggle always-on-top |
| Expand | `Cmd+Shift+E` | Toggle a taller view for long answers |
| Copy | `Cmd+Shift+C` | Copy the current answer text to the clipboard |

## Tuning

Everything is in `configs/settings.yaml`, no code changes needed:

- `speaker.ignore_similarity_threshold` -- raise if your own voice is
  sometimes still transcribed; lower if you're incorrectly ignored.
- `question_detector.keywords` / `denylist_phrases` -- add topics you care
  about, or phrases that keep falsely triggering.
- `stt.model_size` -- smaller (`base`/`small`) if transcription is too slow
  on your machine; larger (`distil-large-v3`) for better accuracy.
- `llm.backend` -- `cli` (default, uses your Claude Pro subscription, slower)
  vs `api` (needs `ANTHROPIC_API_KEY`, much faster) -- see
  [performance.md](performance.md).
