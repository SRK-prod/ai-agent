from unittest.mock import MagicMock

from meeting_copilot.retrieval.qa_bank import QaBankStore


def _store_with_hit(score: float, threshold: float = 0.80) -> QaBankStore:
    client = MagicMock()
    client.collection_exists.return_value = True
    hit = MagicMock()
    hit.score = score
    hit.payload = {"question": "banked q", "answer": "banked a\nCONFIDENCE: 90", "topic": "aws"}
    client.query_points.return_value.points = [hit]
    store = QaBankStore(client=client)
    store._cfg = store._cfg.model_copy(update={"similarity_threshold": threshold})
    return store


def test_lookup_returns_answer_above_threshold():
    banked = _store_with_hit(score=0.91).lookup([0.0])
    assert banked is not None
    assert banked.answer.startswith("banked a")
    assert banked.score == 0.91


def test_lookup_rejects_below_threshold():
    assert _store_with_hit(score=0.71).lookup([0.0]) is None


def test_lookup_handles_missing_collection():
    client = MagicMock()
    client.collection_exists.return_value = False
    assert QaBankStore(client=client).lookup([0.0]) is None


def test_lookup_handles_empty_bank():
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.return_value.points = []
    assert QaBankStore(client=client).lookup([0.0]) is None
