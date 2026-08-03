from meeting_copilot.knowledge.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_tokens=100, overlap_tokens=10) == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("hello world", chunk_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1
    assert "hello world" in chunks[0]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    # consecutive chunks should share some trailing/leading tokens due to overlap
    assert any(
        chunks[i].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1)
    )


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_tokens=10, overlap_tokens=10)
