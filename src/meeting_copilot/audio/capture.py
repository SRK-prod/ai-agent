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
from meeting_copilot.pipeline.events import AudioFrame
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


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
        self._start_time: float = 0.0
        self._callback_count: int = 0

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"Audio capture status: {status}")
        self._callback_count += 1
        if self._callback_count % 64 == 1:  # ~every 2s at 30ms blocks -- confirms the
            # stream is alive and shows real signal level without flooding the log
            peak = float(np.max(np.abs(indata))) if indata.size else 0.0
            logger.debug(f"Audio callback #{self._callback_count}: peak={peak:.4f}")
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

    async def frames(self) -> AsyncIterator[AudioFrame]:
        self._loop = asyncio.get_running_loop()
        self._start_time = time.monotonic()
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
