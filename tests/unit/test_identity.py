import numpy as np

from meeting_copilot.config import SpeakerConfig
from meeting_copilot.speaker.identity import EnrollmentStore, SpeakerIdentity, cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_zero_vector_is_safe():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(a, b) == 0.0


def _embedding(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def test_new_speakers_get_sequential_labels(tmp_path):
    store = EnrollmentStore(db_path=tmp_path / "enrollment.sqlite3")
    cfg = SpeakerConfig(ignore_similarity_threshold=0.75, new_speaker_similarity_threshold=0.70)
    identity = SpeakerIdentity(config=cfg, store=store)

    label_a, is_me_a, _ = identity.assign_speaker(_embedding(seed=1))
    label_b, is_me_b, _ = identity.assign_speaker(_embedding(seed=2))

    assert label_a == "A"
    assert label_b == "B"
    assert is_me_a is False and is_me_b is False


def test_same_speaker_reuses_label(tmp_path):
    store = EnrollmentStore(db_path=tmp_path / "enrollment.sqlite3")
    cfg = SpeakerConfig(ignore_similarity_threshold=0.75, new_speaker_similarity_threshold=0.70)
    identity = SpeakerIdentity(config=cfg, store=store)

    embedding = _embedding(seed=5)
    label_first, _, _ = identity.assign_speaker(embedding)
    label_second, _, _ = identity.assign_speaker(embedding)  # same voice again

    assert label_first == label_second == "A"


def test_enrolled_voice_is_ignored(tmp_path):
    store = EnrollmentStore(db_path=tmp_path / "enrollment.sqlite3")
    me_embedding = _embedding(seed=42)
    store.save(me_embedding, embedding_model="pyannote/embedding")

    cfg = SpeakerConfig(ignore_similarity_threshold=0.75, new_speaker_similarity_threshold=0.70)
    identity = SpeakerIdentity(config=cfg, store=store)

    label, is_me, similarity = identity.assign_speaker(me_embedding)
    assert is_me is True
    assert label == "me"
    assert similarity > 0.99
