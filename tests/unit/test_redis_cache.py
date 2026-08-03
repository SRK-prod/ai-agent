import pytest
from fakeredis.aioredis import FakeRedis

from meeting_copilot.cache.redis_cache import RedisCache


@pytest.fixture
def cache() -> RedisCache:
    # decode_responses=True to match how RedisCache's real client is configured
    return RedisCache(client=FakeRedis(decode_responses=True))


async def test_embedding_cache_miss_then_hit(cache):
    assert await cache.get_embedding("model-x", "hello") is None
    await cache.set_embedding("model-x", "hello", [0.1, 0.2, 0.3])
    assert await cache.get_embedding("model-x", "hello") == [0.1, 0.2, 0.3]


async def test_embedding_cache_is_keyed_by_model(cache):
    await cache.set_embedding("model-a", "hello", [1.0])
    assert await cache.get_embedding("model-b", "hello") is None


async def test_llm_response_cache_roundtrip(cache):
    question = "What's the tradeoff between Kafka and Redis here?"
    chunks = ["chunk one text", "chunk two text"]
    assert await cache.get_llm_response(question, chunks) is None
    await cache.set_llm_response(question, chunks, "Use Kafka for durability.")
    assert await cache.get_llm_response(question, chunks) == "Use Kafka for durability."


async def test_stt_dedup_flags_repeat_utterance(cache):
    assert await cache.seen_recently("speaker_a", "same sentence") is False
    assert await cache.seen_recently("speaker_a", "same sentence") is True


async def test_stt_dedup_is_per_speaker(cache):
    assert await cache.seen_recently("speaker_a", "hello") is False
    assert await cache.seen_recently("speaker_b", "hello") is False
