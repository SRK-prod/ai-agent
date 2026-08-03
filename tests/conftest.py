from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sine_wave() -> np.ndarray:
    """1 second of a 220Hz tone at 16kHz, as a stand-in for real speech audio."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(16000, dtype=np.float32)
