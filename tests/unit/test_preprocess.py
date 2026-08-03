import numpy as np

from meeting_copilot.audio.preprocess import normalize_peak, remove_dc_offset


def test_remove_dc_offset_centers_signal():
    samples = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    result = remove_dc_offset(samples)
    assert np.allclose(result, 0.0)


def test_remove_dc_offset_preserves_shape(sine_wave):
    result = remove_dc_offset(sine_wave)
    assert result.shape == sine_wave.shape


def test_normalize_peak_scales_to_target():
    samples = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    result = normalize_peak(samples, target_peak=0.95)
    assert np.isclose(np.max(np.abs(result)), 0.95, atol=1e-4)


def test_normalize_peak_handles_silence():
    samples = np.zeros(10, dtype=np.float32)
    result = normalize_peak(samples)
    assert np.allclose(result, 0.0)
