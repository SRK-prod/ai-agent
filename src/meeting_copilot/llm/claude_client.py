"""Claude client with two swappable backends (configs/settings.yaml `llm.backend`):

- "cli" (default): shells out to the Claude Code CLI (`claude -p ...`), which
  authenticates via your existing Claude Pro/Max subscription login
  (`claude login`) -- no Anthropic API billing. Tradeoff: each call pays CLI
  process-startup + inference cost, realistically 1-3s+, well past the
  spec's <400ms LLM-latency target (see docs/performance.md). Persistent
  context is approximated via `--resume <session_id>` across calls in a
  meeting, rather than a truly warm connection.
- "api": direct Anthropic Messages API via ANTHROPIC_API_KEY, which is what
  actually hits the <400ms target and supports real prompt caching.

Both are hidden behind ClaudeClient so the rest of the pipeline
(llm/answer_optimizer.py, knowledge/ingestion.py) never needs to know which
one is active.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Protocol

from meeting_copilot.config import LlmConfig, get_config
from meeting_copilot.llm.prompt_templates import CACHE_BREAKPOINT
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

_RETRYABLE_ATTEMPTS = 3  # 1 initial + 2 retries
_RETRY_BACKOFF_SECONDS = (0.5, 1.5)


async def _stream_with_retry(
    make_stream, *args, **kwargs
) -> AsyncIterator[str]:
    """Retries a streaming call, but ONLY before any chunk has been yielded -- that's
    where transient failures (rate limits, connection blips, momentary 5xx) actually
    happen in practice. A mid-stream failure after real content already reached the
    candidate is not retried (retrying would duplicate/corrupt what's already shown),
    so this doesn't compromise the fast-first-token behavior once a stream is flowing.
    Measured live: a burst of concurrent OpenAI calls transiently failed 2/5 times with
    no retry at all -- a single network blip mid-interview should not silently drop the
    only answer the candidate has for that question."""
    last_exc: Exception | None = None
    for attempt in range(_RETRYABLE_ATTEMPTS):
        try:
            first = True
            async for chunk in make_stream(*args, **kwargs):
                first = False
                yield chunk
            return
        except Exception as e:
            if not first:
                raise  # already yielded real content -- don't retry, don't duplicate
            last_exc = e
            if attempt < _RETRYABLE_ATTEMPTS - 1:
                delay = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                logger.warning(
                    f"LLM stream failed before any output (attempt {attempt + 1}/"
                    f"{_RETRYABLE_ATTEMPTS}): {type(e).__name__}: {e} -- retrying in {delay}s"
                )
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


class _ClaudeBackend(Protocol):
    async def complete(self, prompt: str, system: str | None = None) -> str: ...
    def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]: ...


class ClaudeCliBackend:
    def __init__(self, config: LlmConfig):
        self._cfg = config
        self._session_id: str | None = None

    async def complete(self, prompt: str, system: str | None = None) -> str:
        cmd = [self._cfg.cli_binary, "-p", prompt, "--output-format", "json"]
        if system:
            # Text-only backend: the cache breakpoint is an API-block concept, strip it.
            cmd += ["--append-system-prompt", system.replace(CACHE_BREAKPOINT, "\n\n")]
        if self._session_id:
            cmd += ["--resume", self._session_id]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._cfg.cli_timeout_seconds
            )
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"`{self._cfg.cli_binary}` CLI timed out after "
                                f"{self._cfg.cli_timeout_seconds}s") from None

        payload = None
        try:
            payload = json.loads(stdout.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        if proc.returncode != 0 or (payload and payload.get("is_error")):
            # `--output-format json` puts the actual error (e.g. "Not logged in --
            # Please run /login") in stdout's `result` field, not stderr.
            detail = (payload or {}).get("result") or stderr.decode(errors="ignore").strip()
            raise RuntimeError(
                f"`{self._cfg.cli_binary}` CLI failed (exit {proc.returncode}): {detail}"
            )

        assert payload is not None
        self._session_id = payload.get("session_id", self._session_id)
        return (payload.get("result") or "").strip()

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        cmd = [self._cfg.cli_binary, "-p", prompt, "--output-format", "stream-json"]
        if system:
            # Text-only backend: the cache breakpoint is an API-block concept, strip it.
            cmd += ["--append-system-prompt", system.replace(CACHE_BREAKPOINT, "\n\n")]
        if self._session_id:
            cmd += ["--resume", self._session_id]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        assert proc.stdout is not None
        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode(errors="ignore").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "system" and event.get("session_id"):
                    self._session_id = event["session_id"]

                delta = event.get("delta", {})
                text = delta.get("text") if isinstance(delta, dict) else None
                if text:
                    yield text
        finally:
            await proc.wait()


class ClaudeApiBackend:
    def __init__(self, config: LlmConfig):
        import anthropic

        self._anthropic = anthropic
        self._cfg = config
        api_key = get_config().secrets.require_anthropic_key()
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def _system_blocks(self, system: str | None):
        if not system:
            return self._anthropic.NOT_GIVEN  # omit the param entirely, not None
        # cache_control lets repeated calls in the same meeting reuse the (large,
        # unchanging) persona/instructions prefix instead of re-billing for it. The
        # breakpoint MUST sit between the static prefix and the per-question tail: a single
        # block spanning both is a cache key that changes with every question category, so
        # nothing ever hits (measured 2026-08-24: cache_read=0 on every call, ~21K tokens
        # re-billed each time; after the split, cache_read≈18.9K on every call but the first).
        static, marker, tail = system.partition(CACHE_BREAKPOINT)
        if not marker:  # no breakpoint (e.g. a caller-supplied prompt) -- cache it whole
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return [
            {"type": "text", "text": static, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": tail},
        ]

    async def complete(self, prompt: str, system: str | None = None) -> str:
        last_exc: Exception | None = None
        for attempt in range(_RETRYABLE_ATTEMPTS):
            try:
                response = await self._client.messages.create(
                    model=self._cfg.model,
                    max_tokens=self._cfg.max_tokens,
                    system=self._system_blocks(system),
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
            except Exception as e:
                last_exc = e
                if attempt < _RETRYABLE_ATTEMPTS - 1:
                    delay = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(f"Claude complete() failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def _raw_stream(self, prompt: str, system: str | None) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        return _stream_with_retry(self._raw_stream, prompt, system)


class OpenAiApiBackend:
    """ChatGPT backend via OpenAI's Chat Completions API -- same _ClaudeBackend interface,
    so ClaudeClient/orchestrator/answer_optimizer don't need to know which provider is
    actually generating the answer. Added as a fallback option: if this backend errors,
    switch configs/settings.yaml llm.backend back to "api" (Claude) immediately."""

    def __init__(self, config: LlmConfig):
        from openai import AsyncOpenAI

        self._cfg = config
        api_key = get_config().secrets.require_openai_key()
        self._client = AsyncOpenAI(api_key=api_key)

    def _messages(self, prompt: str, system: str | None):
        messages = []
        if system:
            # Text-only backend: the cache breakpoint is an API-block concept, strip it.
            messages.append(
                {"role": "system", "content": system.replace(CACHE_BREAKPOINT, "\n\n")}
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    async def complete(self, prompt: str, system: str | None = None) -> str:
        last_exc: Exception | None = None
        for attempt in range(_RETRYABLE_ATTEMPTS):
            try:
                response = await self._client.chat.completions.create(
                    model=self._cfg.openai_model,
                    max_tokens=self._cfg.max_tokens,
                    messages=self._messages(prompt, system),
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_exc = e
                if attempt < _RETRYABLE_ATTEMPTS - 1:
                    delay = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(f"OpenAI complete() failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def _raw_stream(self, prompt: str, system: str | None) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._cfg.openai_model,
            max_tokens=self._cfg.max_tokens,
            messages=self._messages(prompt, system),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        return _stream_with_retry(self._raw_stream, prompt, system)


class ClaudeClient:
    """Facade: picks the backend from configs/settings.yaml `llm.backend` once, at construction."""

    def __init__(self, config: LlmConfig | None = None):
        self._cfg = config or get_config().llm
        self._backend: _ClaudeBackend
        if self._cfg.backend == "cli":
            self._backend = ClaudeCliBackend(self._cfg)
        elif self._cfg.backend == "api":
            self._backend = ClaudeApiBackend(self._cfg)
        elif self._cfg.backend == "openai":
            self._backend = OpenAiApiBackend(self._cfg)
        else:
            raise ValueError(f"Unknown llm.backend: {self._cfg.backend!r}")
        logger.info(f"ClaudeClient using backend={self._cfg.backend}")

    async def complete(self, prompt: str, system: str | None = None) -> str:
        return await self._backend.complete(prompt, system)

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        async for chunk in self._backend.stream(prompt, system):
            yield chunk
