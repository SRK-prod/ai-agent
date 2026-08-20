# meeting-copilot

A desktop agent (macOS and Windows) that listens continuously to meeting
audio, learns and ignores your own voice, transcribes everyone else, detects
when someone asks a technical question, retrieves relevant context from a
pre-built knowledge base, and shows a concise Claude-generated answer in a
floating always-on-top overlay. No manual interaction is required once the
meeting starts.

See [`docs/architecture.md`](docs/architecture.md) for how the pipeline fits
together, [`docs/installation.md`](docs/installation.md) for full macOS setup,
and [`docs/installation-windows.md`](docs/installation-windows.md) for
Windows (system audio routing, models, credentials).

## Quickstart (macOS)

```bash
# 1. System prerequisites (see docs/installation.md for details)
brew install ffmpeg portaudio blackhole-2ch

# 2. Python env + deps
make setup

# 3. Backing services (Qdrant + Redis)
make services

# 4. Copy env template and fill in secrets you have
cp .env.example .env

# 5. One-time: enroll your voice (30-60s recording)
make enroll

# 6. One-time (and whenever configs/topics.yaml changes): build the knowledge base
make ingest

# 7. Run
make run
```

## Quickstart (Windows)

See [`docs/installation-windows.md`](docs/installation-windows.md) for the
full walkthrough (VB-Audio Virtual Cable setup, venv, hotkeys). Once set up:

```powershell
scripts\start.ps1
```

## LLM backend

By default this project talks to Claude through the direct **Messages API**
(`llm.backend: api`, model `claude-haiku-4-5-20251001` -- fast and cheap),
using `ANTHROPIC_API_KEY` in `.env` (console.anthropic.com, separate from a
Claude Pro/Max subscription). Set `llm.backend: cli` in
`configs/settings.yaml` instead to use your Claude Pro/Max subscription via
the Claude Code CLI (`claude login`) with no API billing, at the cost of
much slower answers -- see [`docs/performance.md`](docs/performance.md) for
the measured latency tradeoff.

## Testing

```bash
make test-fast   # unit + mocked-integration tests, no models/keys required
make test        # everything, including slow tests that use real models
```
