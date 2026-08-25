"""Regression tests for AudioCapture.health() -- see the module docstring in
audio/capture.py for the critical fact these all exist to protect: this stream carries
ONLY the interviewer's side, so long silence while the candidate answers is normal and
must never by itself read as AUDIO_INPUT_LOST. Only infrastructure evidence (callback
stall, device gone/changed) may produce that state.

None of these touch a real audio device -- AudioCapture is constructed with
input_device=None (skips device resolution entirely) and health-relevant timestamps are
poked directly, the same pattern used to verify this live before landing it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from meeting_copilot.audio import capture as capture_module
from meeting_copilot.config import AudioConfig


def _make_capture(input_device: str | None = None) -> capture_module.AudioCapture:
    cfg = AudioConfig(sample_rate=16000, channels=1, block_ms=30, input_device=input_device)
    return capture_module.AudioCapture(cfg)


class _FakeLoop:
    def call_soon_threadsafe(self, fn, arg):
        fn(arg)


def test_normal_long_silence_stays_silent_not_lost():
    """Test 1 -- callbacks/buffers healthy, RMS=peak=0 for 120s -> AUDIO_SILENT, never
    AUDIO_INPUT_LOST."""
    c = _make_capture()
    now = time.monotonic()
    c._last_callback_at = now  # callback just fired -- infrastructure is alive
    c._last_nonzero_signal_at = now - 120.0  # but silent for 2 full minutes
    c._last_peak = 0.0
    c._last_rms = 0.0

    health = c.health()

    assert health.state == "AUDIO_SILENT"
    assert health.reason == "NO_INTERVIEWER_SPEECH"


def test_long_candidate_answer_round_trips_through_silent():
    """Test 2 -- interviewer speaks, 120s of candidate-answering silence, interviewer
    speaks again. Expected: AUDIO_ACTIVE -> AUDIO_SILENT -> AUDIO_ACTIVE, no
    AUDIO_INPUT_LOST anywhere in the sequence."""
    c = _make_capture()
    now = time.monotonic()

    c._last_callback_at = now
    c._last_nonzero_signal_at = now
    assert c.health().state == "AUDIO_ACTIVE"

    c._last_callback_at = now  # callback keeps firing throughout -- infra never breaks
    c._last_nonzero_signal_at = now - 120.0
    mid = c.health()
    assert mid.state == "AUDIO_SILENT"
    assert mid.reason == "NO_INTERVIEWER_SPEECH"

    c._loop = _FakeLoop()
    c._start_time = now
    c._callback(np.array([[0.05]], dtype=np.float32), 1, None, None)
    assert c.health().state == "AUDIO_ACTIVE"


def test_callback_stall_is_input_lost():
    """Test 3 -- callbacks were arriving, then stop. Recent signal is irrelevant once the
    callback itself has stalled -- that alone must mean AUDIO_INPUT_LOST."""
    c = _make_capture()
    now = time.monotonic()
    c._last_callback_at = now - 10.0  # well past _CALLBACK_STALL_SECONDS
    c._last_nonzero_signal_at = now  # signal was fine moments ago -- must not matter

    health = c.health()

    assert health.state == "AUDIO_INPUT_LOST"
    assert health.reason == "CALLBACK_STALLED"


def test_device_disappears_is_input_lost(monkeypatch):
    """Test 4 -- the configured device index no longer has any input channels (or is gone
    from the enumerated list) -> AUDIO_INPUT_LOST/DEVICE_UNAVAILABLE."""
    c = _make_capture()
    c._device = 0
    c._expected_device_name = "BlackHole"
    now = time.monotonic()
    c._last_callback_at = now
    c._last_nonzero_signal_at = now

    monkeypatch.setattr(
        capture_module.sd, "query_devices", lambda: [{"name": "BlackHole", "max_input_channels": 0}]
    )

    health = c.health()

    assert health.state == "AUDIO_INPUT_LOST"
    assert health.reason == "DEVICE_UNAVAILABLE"


def test_device_changed_is_input_lost(monkeypatch):
    """Test 5 -- the index we opened now belongs to a differently-named device (CoreAudio
    renumbering) -> AUDIO_INPUT_LOST/DEVICE_CHANGED."""
    c = _make_capture()
    c._device = 0
    c._expected_device_name = "BlackHole"
    now = time.monotonic()
    c._last_callback_at = now
    c._last_nonzero_signal_at = now

    monkeypatch.setattr(
        capture_module.sd,
        "query_devices",
        lambda: [{"name": "MacBook Pro Microphone", "max_input_channels": 1}],
    )

    health = c.health()

    assert health.state == "AUDIO_INPUT_LOST"
    assert health.reason == "DEVICE_CHANGED"


def test_recovery_from_callback_stall_through_real_callback():
    """Test 6 -- after a genuine callback stall (AUDIO_INPUT_LOST), the real _callback()
    path firing again must bring it back to AUDIO_ACTIVE."""
    c = _make_capture()
    now = time.monotonic()
    c._last_callback_at = now - 10.0
    c._last_nonzero_signal_at = now - 10.0
    assert c.health().state == "AUDIO_INPUT_LOST"

    c._loop = _FakeLoop()
    c._start_time = now
    c._callback(np.array([[0.05]], dtype=np.float32), 1, None, None)

    assert c.health().state == "AUDIO_ACTIVE"


def test_device_check_skipped_when_no_device_configured(monkeypatch):
    """When no specific input_device is configured (self._device is None -- use system
    default), there is nothing to compare against, so the device check must never itself
    produce a failure."""
    c = _make_capture(input_device=None)
    assert c._device is None
    now = time.monotonic()
    c._last_callback_at = now
    c._last_nonzero_signal_at = now

    monkeypatch.setattr(
        capture_module.sd, "query_devices", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    health = c.health()

    assert health.state == "AUDIO_ACTIVE"
