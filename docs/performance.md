# Performance

## Targets (from the original spec)

| Stage | Target |
|---|---|
| Speech detection (VAD) | <50ms |
| STT | <200ms |
| Retrieval | <50ms |
| LLM | <400ms |
| Overlay render | <50ms |
| **Total** | **<800ms** |

These are tracked per-stage via `meeting_copilot_stage_latency_seconds{stage=...}`
(Prometheus histogram, `pipeline/metrics.py`), viewable in Grafana
(`docker compose --profile monitoring up -d`).

## The LLM backend tradeoff (read this before tuning anything else)

**Default is now `llm.backend: api`** (direct Anthropic Messages API,
model `claude-haiku-4-5-20251001` -- fastest/cheapest current Claude model,
plenty for synthesizing retrieved context into a short answer). This is the
path the <400ms LLM / <800ms total targets were designed around: streamed
tokens, prompt caching, a warm HTTPS connection. Needs `ANTHROPIC_API_KEY`
billing in `.env` (console.anthropic.com -- separate from a Claude Pro/Max
subscription).

**`llm.backend: cli`** (uses your Claude Pro/Max subscription via the Claude
Code CLI, no separate API billing) is the alternative if you'd rather not
pay for API usage. Each call pays CLI process startup + full model inference
before returning anything -- **measured 1-3 seconds for a short answer, up
to ~106 seconds for a long/thorough generation** (a ~4500-token "research
notes" ingestion prompt). Not a bug, just what a subprocess-per-call design
costs. `llm.cli_timeout_seconds` defaults to 180s to cover the slow case.
Switch back to it in `configs/settings.yaml` if avoiding API billing matters
more than latency for your use case.

## Where the rest of the budget actually goes

- **STT**: dominated by `stt.model_size`. `distil-large-v3` (default) is
  more accurate but slower than `base`/`small` on CPU. Benchmark on your own
  machine with `make download-models` then a manual `FasterWhisperEngine`
  call before committing to a size.
- **Speaker ID**: one pyannote embedding inference per utterance
  (`speaker.window_seconds`, default 5s chunks) -- shrink the window for
  faster-but-noisier identification.
- **Retrieval**: one local embedding call for the question (unless cached,
  CPU-bound, no network) plus a Qdrant query. Both should comfortably fit
  the <50ms target; check `configs/settings.yaml` `retrieval.top_k_vector`
  if Qdrant itself is slow (fewer candidates = faster, less recall).
- **Caching**: repeated questions in the same meeting skip retrieval+LLM
  entirely via the Redis-backed LLM response cache
  (`cache.llm_response_ttl_seconds`).

## Benchmarking

```bash
make test          # includes tests/load/ pytest-benchmark throughput checks
```

`tests/load/` currently covers the offline ingestion chunking path; add
further `pytest-benchmark` cases there as you tune STT/retrieval on your own
hardware.
