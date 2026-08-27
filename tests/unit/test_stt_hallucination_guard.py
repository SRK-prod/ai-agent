import numpy as np
import pytest
from fakeredis.aioredis import FakeRedis

from meeting_copilot.cache.redis_cache import RedisCache
from meeting_copilot.pipeline.events import DiarizedSegment, SpeechSegment
from meeting_copilot.stt.faster_whisper_engine import SttStage, _is_hallucinated, _is_near_silent


def _segment(samples: np.ndarray, *, is_me: bool = False) -> DiarizedSegment:
    speech = SpeechSegment(samples=samples, sample_rate=16000, start_time=0.0, end_time=1.0)
    return DiarizedSegment(segment=speech, speaker_id="speaker_a", is_me=is_me, similarity_to_me=0.1)


class _FakeEngine:
    """Records whether transcribe_samples was ever called -- the near-silence gate must
    prevent this from firing at all, not just discard its result afterward."""

    def __init__(self, text: str = "hello"):
        self.called = False
        self._text = text
        self._cfg = type("Cfg", (), {"language": "en"})()

    def transcribe_samples(self, samples, sample_rate) -> str:
        self.called = True
        return self._text


@pytest.fixture
def cache() -> RedisCache:
    return RedisCache(client=FakeRedis(decode_responses=True))


# --- _is_hallucinated: existing single-word repetition loop (must keep working) ---


def test_single_word_repetition_loop_is_flagged():
    assert _is_hallucinated("create create create create create create") is True


def test_real_speech_is_never_flagged():
    assert _is_hallucinated(
        "How would you design a highly available AWS architecture for a financial platform"
    ) is False


def test_short_real_answer_is_never_flagged():
    assert _is_hallucinated("What is Terraform") is False


# --- _is_hallucinated: new short-phrase repetition loop (2026-08-26 fix) ---


def test_three_word_phrase_repetition_loop_is_flagged():
    # Captured verbatim from a real interview-call test session.
    text = "the other one, " * 12
    assert _is_hallucinated(text) is True


def test_five_word_phrase_repetition_loop_is_flagged():
    # Captured verbatim from a real interview-call test session.
    text = "I could be red, or " * 8
    assert _is_hallucinated(text) is True


def test_short_text_is_never_flagged_even_if_repetitive():
    # The <6-word floor must still apply -- a real short answer like "yes yes I agree"
    # should never be discarded.
    assert _is_hallucinated("yes yes I agree") is False


# --- _is_near_silent ---


def test_pure_silence_is_near_silent():
    assert _is_near_silent(np.zeros(32000, dtype=np.float32)) is True


def test_tiny_noise_floor_is_near_silent():
    samples = (np.random.default_rng(0).standard_normal(32000) * 0.001).astype(np.float32)
    assert _is_near_silent(samples) is True


def test_real_level_speech_is_not_near_silent():
    samples = (np.random.default_rng(0).standard_normal(32000) * 0.05).astype(np.float32)
    assert _is_near_silent(samples) is False


def test_empty_array_is_near_silent():
    assert _is_near_silent(np.array([], dtype=np.float32)) is True


# --- SttStage: the gate must short-circuit before the expensive engine call ---


async def test_near_silent_segment_never_reaches_the_engine(cache):
    engine = _FakeEngine()
    stage = SttStage(engine=engine, cache=cache)

    result = await stage.transcribe(_segment(np.zeros(32000, dtype=np.float32)))

    assert result is None
    assert engine.called is False


async def test_real_audio_still_reaches_the_engine(cache):
    engine = _FakeEngine(text="how would you design a highly available system")
    stage = SttStage(engine=engine, cache=cache)
    samples = (np.random.default_rng(0).standard_normal(32000) * 0.05).astype(np.float32)

    result = await stage.transcribe(_segment(samples))

    assert engine.called is True
    assert result is not None
    assert result.text == "how would you design a highly available system"
