# Developer Guide

## Layout

```
src/meeting_copilot/
  config.py        pydantic-settings: configs/settings.yaml (tunables) + .env (secrets)
  paths.py          all filesystem paths in one place
  audio/            capture (sounddevice/CoreAudio), preprocess (noise/DC-offset)
  vad/              Silero VAD -> SpeechSegment assembly
  speaker/          enrollment (CLI recorder), diarization (per-segment embedding),
                    identity (SQLite store + online speaker clustering)
  stt/              Faster-Whisper transcription
  nlp/               question_detector (rule-based, swappable via QuestionDetector ABC)
  knowledge/        chunking, local sentence-transformers embeddings, ingestion (topics.yaml -> Qdrant),
                    store (SQLite ingestion-job log)
  retrieval/        Qdrant client, hybrid (semantic + keyword) search
  llm/               ClaudeClient (cli/api backends), prompts, answer optimizer
  cache/            Redis: embedding cache, LLM response cache, STT dedup
  pipeline/         events.py (typed stage contracts), orchestrator.py (wiring),
                    metrics.py (Prometheus histograms)
  server/           FastAPI app + WebSocket + entrypoint
  desktop/          PySide6 overlay, global hotkeys, app entrypoint
  utils/            loguru setup
```

## Config system

Everything non-secret lives in `configs/settings.yaml`, loaded once into a
typed `AppConfig` (see `config.py`) and cached via `get_config()`. Secrets
(`ANTHROPIC_API_KEY`, `HF_TOKEN`, service URLs) come from
`.env`/environment via the `Secrets` `pydantic-settings` model, with
`require_*` helpers that raise a `MissingCredentialError` pointing back at
this doc when something's missing. Tests that need different config just
construct the relevant `*Config` model directly rather than mutating global
state.

## Adding a new pipeline stage

The pipeline is a strict chain of typed events in `pipeline/events.py`
(`AudioFrame -> SpeechSegment -> DiarizedSegment -> Transcript ->
DetectedQuestion -> RetrievedContext -> Answer`). To add or replace a stage:

1. Add/extend the dataclass in `events.py` if the contract needs a new field.
2. Implement the stage as a small class with one clear async (or
   thread-offloaded sync) method -- follow `stt/faster_whisper_engine.py`'s
   `SttStage` pattern: a thin pipeline-facing wrapper around a lower-level
   engine class, so the engine itself stays independently testable.
2. Wire it into `pipeline/orchestrator.py::MeetingPipeline._handle_segment`.
3. Wrap it in a `StageTimer("your_stage_name")` (see `pipeline/metrics.py`)
   so it shows up in Prometheus/Grafana automatically.

## Testing conventions

- `tests/unit/` -- pure-logic tests, no models/network/services. Should run
  in well under a second each.
- `tests/integration/` -- multiple modules composed together. Use fakes
  (see `tests/integration/test_pipeline_slice.py`'s `FakeQdrantStore`/
  `FakeEmbedder`) rather than a real Qdrant/embedding model so these stay CI-safe.
  Anything that genuinely needs a real ML model or hardware, mark
  `@pytest.mark.slow`.
- `tests/load/` -- `pytest-benchmark` throughput checks for anything on the
  offline ingestion path where a regression would make `make ingest` slow.
- `tests/e2e` markers (via `@pytest.mark.e2e`) are for the one Playwright
  browser-meeting path described in `docs/architecture.md` -- needs real
  BlackHole routing on the host, run manually, not in CI.

Run `make test-fast` (excludes `slow`/`e2e`) during normal development;
`make test` for everything.

## Swapping the LLM backend

`llm/claude_client.py::ClaudeClient` picks `ClaudeCliBackend` or
`ClaudeApiBackend` based on `configs/settings.yaml` `llm.backend`. Both
implement the same `complete()`/`stream()` interface, so nothing else in the
codebase needs to change when switching.
