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

# Audio health (see AudioCapture.health()) -- observability only, no automatic recovery yet.
#
# CRITICAL DESIGN FACT, learned from real interview telemetry 2026-08-25: this capture
# stream carries ONLY the interviewer's side (BlackHole/system audio), never the
# candidate's own mic. While the candidate is answering -- routinely 30s to several
# minutes -- this stream is 100% silent by design. That is normal conversation, not a
# failure, so DURATION OF ZERO SIGNAL MUST NEVER BY ITSELF ESCALATE TO A FAILURE STATE.
# An earlier version of this watchdog did exactly that (zero signal for 20s -> "input
# lost") and produced a false red warning on nearly every candidate answer -- confirmed
# live: 7 false "AUDIO_INPUT_LOST" transitions in one ~11-minute session, every single one
# recovering on its own the moment the interviewer spoke again. Silence is not failure.
#
# So AUDIO_INPUT_LOST is now driven ONLY by evidence the CAPTURE INFRASTRUCTURE itself is
# broken -- the callback stopped firing, or the expected device disappeared/changed --
# never by how long the signal has been quiet. AUDIO_ACTIVE vs AUDIO_SILENT is purely
# "is there signal right now", informational either way.
_SIGNAL_PEAK_THRESHOLD = 0.0005  # a buffer at or below this peak counts as "silent"

# Pure UI smoothing between AUDIO_ACTIVE and AUDIO_SILENT -- NOT a failure threshold, and
# never escalates to AUDIO_INPUT_LOST. Without this, the indicator would flicker
# active/silent on every ~100-300ms gap between words in the interviewer's own normal
# speech. Kept short and deliberately far below any duration that could be mistaken for a
# failure signal.
_RECENT_SIGNAL_WINDOW_SECONDS = 1.5

# If the callback itself hasn't fired in this long, the stream has stopped delivering
# buffers entirely -- this, not silence, is the primary infrastructure-failure signal.
# Kept short: at block_ms=30 a healthy stream calls back roughly every 30ms, so several
# seconds of total callback silence is already a clear stall, not scheduler jitter, and is
# completely independent of whether the interviewer is currently talking.
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
        # The configured (partial, case-insensitive) name pattern -- kept so health() can
        # later re-check that the device now sitting at self._device's index is still the
        # SAME physical device, not one the OS silently reassigned that index to (e.g. after
        # a Bluetooth device connects/disconnects and CoreAudio renumbers everything).
        self._expected_device_name = self._config.input_device
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

    def _check_device(self) -> str | None:
        """None if the expected input device is still present and unchanged; otherwise
        "DEVICE_UNAVAILABLE" (gone, or no longer has input channels) or "DEVICE_CHANGED"
        (the index we opened now belongs to a different-named device -- CoreAudio can
        renumber devices when one connects/disconnects, e.g. Bluetooth or an Aggregate/
        Multi-Output device changing). Enumerates devices fresh each call -- cheap metadata,
        deliberately kept off the realtime audio callback and only called from the async
        watchdog poll (every few seconds), never from the audio thread.
        """
        if self._device is None:
            return None  # no specific device was configured -- nothing to compare against
        try:
            devices = sd.query_devices()
        except Exception:
            # A transient enumeration hiccup shouldn't itself masquerade as a device
            # failure -- the callback-liveness check above is the reliable signal for that.
            return None
        if self._device >= len(devices) or devices[self._device]["max_input_channels"] <= 0:
            return "DEVICE_UNAVAILABLE"
        current_name = devices[self._device]["name"]
        if self._expected_device_name and self._expected_device_name.lower() not in current_name.lower():
            return "DEVICE_CHANGED"
        return None

    def health(self) -> AudioHealth:
        """Point-in-time read of whether the CAPTURE INFRASTRUCTURE is actually working --
        does NOT block or touch the stream itself, safe to poll from a separate watchdog
        loop on a timer.

        Deliberately answers "can this still hear the interviewer if they speak", not "has
        the interviewer spoken recently" -- see the module docstring above for why those are
        different questions and why conflating them produced false alarms. AUDIO_INPUT_LOST
        requires actual evidence the pipeline is broken (callback stalled, device gone/
        changed); AUDIO_ACTIVE vs AUDIO_SILENT is purely "is there signal right now" and
        never escalates on its own no matter how long the silence lasts.
        """
        now = time.monotonic()
        since_callback = now - self._last_callback_at
        since_signal = now - self._last_nonzero_signal_at
        if since_callback > _CALLBACK_STALL_SECONDS:
            state, reason = "AUDIO_INPUT_LOST", "CALLBACK_STALLED"
        else:
            device_reason = self._check_device()
            if device_reason is not None:
                state, reason = "AUDIO_INPUT_LOST", device_reason
            elif since_signal <= _RECENT_SIGNAL_WINDOW_SECONDS:
                state, reason = "AUDIO_ACTIVE", None
            else:
                state, reason = "AUDIO_SILENT", "NO_INTERVIEWER_SPEECH"
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
