from meeting_copilot.retrieval.hybrid_search import _keyword_overlap_score


def test_full_overlap_scores_one():
    query_terms = {"kafka", "partitions"}
    assert _keyword_overlap_score(query_terms, "kafka partitions explained") == 1.0


def test_no_overlap_scores_zero():
    query_terms = {"kafka", "partitions"}
    assert _keyword_overlap_score(query_terms, "totally unrelated text") == 0.0


def test_partial_overlap_is_fractional():
    query_terms = {"kafka", "partitions", "replication"}
    score = _keyword_overlap_score(query_terms, "kafka partitions are neat")
    assert 0.0 < score < 1.0


def test_empty_query_terms_scores_zero():
    assert _keyword_overlap_score(set(), "some chunk text") == 0.0
