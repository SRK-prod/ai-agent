"""Live audio capture via CoreAudio (through PortAudio/sounddevice).

Expects the input device to be a BlackHole-backed Multi-Output Device (or
Loopback) so both the meeting app's output and the user's mic are present in
one stream -- see docs/installation.md for the macOS Audio MIDI Setup steps.
That device routing is a manual, one-time system change; this module just
captures whatever device is configured.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

from meeting_copilot.config import AudioConfig, get_config
from meeting_copilot.pipeline.events import AudioFrame, AudioHealth
from meeting_copilot.utils.logging import get_logger

logger = get_logger()

# Audio health thresholds (see AudioCapture.health()) -- observability only, no automatic
# recovery yet. Deliberately plain module constants, matching every other tunable timing
# value in this codebase (e.g. orchestrator.py's debounce/revision-window constants):
# edit the number, restart. These starting points are from the spec, not measured yet --
# tune them once real interview sessions produce a few actual dropout occurrences.
_SIGNAL_PEAK_THRESHOLD = 0.0005  # a buffer at or below this peak counts as "silent"
_AUDIO_SILENT_AFTER_SECONDS = 10.0  # 0-10s of continuous silence is normal (pauses, etc.)
_AUDIO_LOST_AFTER_SECONDS = 20.0  # 10-20s -> AUDIO_SILENT, 20s+ -> AUDIO_INPUT_LOST
# If the callback itself hasn't fired in this long, the stream/device has stopped
# delivering buffers at all -- a different failure (CALLBACK_STALLED) from "callbacks are
# arriving but every buffer is silent" (ZERO_SIGNAL). Kept short: at block_ms=30 a healthy
# stream calls back roughly every 30ms, so several seconds of total silence from the
# callback itself is already a clear stall, not scheduler jitter.
_CALLBACK_STALL_SECONDS = 5.0


def list_input_devices() -> list[dict]:
    """Enumerate CoreAudio input-capable devices, for picking `audio.input_device`."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "max_input_channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def resolve_device(name: str | None) -> int | None:
    """Look up a CoreAudio input device index by (partial, case-insensitive) name."""
    if not name:
        return None
    for d in list_input_devices():
        if name.lower() in d["name"].lower():
            return d["index"]
    available = ", ".join(d["name"] for d in list_input_devices())
    raise ValueError(f"Input device '{name}' not found. Available devices: {available}")


_resolve_device = resolve_device  # internal alias used within this module


class AudioCapture:
    """Async iterator over AudioFrame, backed by a sounddevice InputStream."""

    def __init__(self, config: AudioConfig | None = None):
        self._config = config or get_config().audio
        self._device = _resolve_device(self._config.input_device)
        self._blocksize = max(1, int(self._config.sample_rate * self._config.block_ms / 1000))
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Initialized at construction (not left at 0.0) so health() reads sane values --
        # "seconds since the stream started" -- even if polled before frames() has actually
        # begun the stream. frames() re-baselines these to the real stream-start instant.
        self._start_time: float = time.monotonic()
        self._callback_count: int = 0
        self._last_callback_at: float = self._start_time
        self._last_nonzero_signal_at: float = self._start_time
        self._last_peak: float = 0.0
        self._last_rms: float = 0.0

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"Audio capture status: {status}")
        now = time.monotonic()
        self._last_callback_at = now
        self._callback_count += 1
        peak = float(np.max(np.abs(indata))) if indata.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
        self._last_peak = peak
        self._last_rms = rms
        if peak > _SIGNAL_PEAK_THRESHOLD:
            self._last_nonzero_signal_at = now
        if self._callback_count % 64 == 1:  # ~every 2s at 30ms blocks -- confirms the
            # stream is alive and shows real signal level without flooding the log
            logger.debug(f"Audio callback #{self._callback_count}: peak={peak:.4f} rms={rms:.4f}")
        frame = AudioFrame(
            samples=indata[:, 0].copy() if indata.ndim > 1 else indata.copy(),
            sample_rate=self._config.sample_rate,
            timestamp=time.monotonic() - self._start_time,
        )
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._enqueue_nowait, frame)

    def _enqueue_nowait(self, frame: AudioFrame) -> None:
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping frame (consumer too slow)")

    def health(self) -> AudioHealth:
        """Point-in-time read of whether the stream is actually delivering usable signal --
        does NOT block or touch the stream itself, safe to poll from a separate watchdog
        loop on a timer. See AudioHealth for what each field means and _AUDIO_*_SECONDS
        above for the thresholds.
        """
        now = time.monotonic()
        since_callback = now - self._last_callback_at
        since_signal = now - self._last_nonzero_signal_at
        if since_callback > _CALLBACK_STALL_SECONDS:
            state, reason = "AUDIO_INPUT_LOST", "CALLBACK_STALLED"
        elif since_signal >= _AUDIO_LOST_AFTER_SECONDS:
            state, reason = "AUDIO_INPUT_LOST", "ZERO_SIGNAL"
        elif since_signal >= _AUDIO_SILENT_AFTER_SECONDS:
            state, reason = "AUDIO_SILENT", "ZERO_SIGNAL"
        else:
            state, reason = "AUDIO_ACTIVE", None
        return AudioHealth(
            state=state,
            reason=reason,
            seconds_since_signal=since_signal,
            seconds_since_callback=since_callback,
            last_peak=self._last_peak,
            last_rms=self._last_rms,
            callback_count=self._callback_count,
        )

    async def frames(self) -> AsyncIterator[AudioFrame]:
        self._loop = asyncio.get_running_loop()
        self._start_time = time.monotonic()
        self._last_callback_at = self._start_time
        self._last_nonzero_signal_at = self._start_time
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        with self._stream:
            logger.info(
                f"Audio capture started: device={self._device}, "
                f"sample_rate={self._config.sample_rate}, block_ms={self._config.block_ms}"
            )
            try:
                while True:
                    yield await self._queue.get()
            finally:
                logger.info("Audio capture stopped")
