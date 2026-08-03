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
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


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
            cmd += ["--append-system-prompt", system]
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
            cmd += ["--append-system-prompt", system]
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
        # unchanging) persona/instructions prefix instead of re-billing for it.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    async def complete(self, prompt: str, system: str | None = None) -> str:
        response = await self._client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class ClaudeClient:
    """Facade: picks the backend from configs/settings.yaml `llm.backend` once, at construction."""

    def __init__(self, config: LlmConfig | None = None):
        self._cfg = config or get_config().llm
        self._backend: _ClaudeBackend
        if self._cfg.backend == "cli":
            self._backend = ClaudeCliBackend(self._cfg)
        elif self._cfg.backend == "api":
            self._backend = ClaudeApiBackend(self._cfg)
        else:
            raise ValueError(f"Unknown llm.backend: {self._cfg.backend!r}")
        logger.info(f"ClaudeClient using backend={self._cfg.backend}")

    async def complete(self, prompt: str, system: str | None = None) -> str:
        return await self._backend.complete(prompt, system)

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        async for chunk in self._backend.stream(prompt, system):
            yield chunk
