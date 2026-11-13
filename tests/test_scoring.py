import copy
import pytest
from memory_rerank.scoring import minmax, rank, available, touch_selected, valid_at
from memory_rerank.metrics import evaluate

NOW = "2026-11-13T12:00:00+08:00"
Q = {"query_time": NOW}


def memory(mid, at, **kwargs):
    return dict(id=mid, content=mid, created_at=at, last_accessed_at=at, importance=5, **kwargs)


def test_recency_is_elapsed_time_and_recent_wins():
    ms = [memory("old", "2026-11-12T12:00:00+08:00"), memory("new", "2026-11-13T11:00:00+08:00")]
    rows = rank(ms, {"old": 0.5, "new": 0.5}, Q, "ga_paper_score")
    assert rows[0]["memory_id"] == "new"
    assert rows[1]["recency_raw"] == pytest.approx(0.995 ** 24)


def test_no_mutation_and_optional_touch():
    ms = [memory("a", "2026-11-12T12:00:00+08:00")]
    original = copy.deepcopy(ms)
    rank(ms, {"a": 0.2}, Q, "ga_paper_score")
    assert ms == original
    touch_selected(ms, ["a"], NOW)
    assert ms[0]["last_accessed_at"] == NOW


def test_future_records_never_enter_candidates():
    ms = [memory("future", "2026-11-14T12:00:00+08:00")]
    assert available(ms, NOW) == []
    assert rank(ms, {}, Q, "semantic") == []


def test_history_query_keeps_old_version():
    old = memory("old", "2026-11-01T00:00:00+08:00", valid_to="2026-11-12T00:00:00+08:00")
    assert not valid_at(old, NOW)
    assert valid_at(old, "2026-11-10T00:00:00+08:00")


def test_constant_empty_and_invalid_scores():
    assert minmax([]) == []
    assert minmax([2, 2]) == [0.5, 0.5]
    with pytest.raises(ValueError): minmax([float("nan")])
    with pytest.raises(ValueError): rank([], {}, Q, "wrong")


def test_underflow_does_not_destroy_recency_order():
    ms = [memory("a", "2000-01-01T00:00:00+00:00"), memory("b", "2001-01-01T00:00:00+00:00")]
    rows = rank(ms, {"a": 0.5, "b": 0.5}, Q, "semantic_recency")
    assert rows[0]["memory_id"] == "b"


def test_metrics_hand_calculated():
    query = {"relevance_grades": {"a": 1, "b": 1}, "obsolete_ids": ["x"]}
    m = evaluate(["x", "a", "b"], query, 2)
    assert m["recall"] == 0.5
    assert m["mrr"] == 0.5
    assert m["obsolete_rate"] == 0.5
    assert m["complete_evidence"] == 0
    assert evaluate(["a", "b"], query, 2)["ndcg"] == 1


def test_no_gold_is_not_perfect_score():
    assert evaluate(["x"], {"relevance_grades": {}}, 1)["recall"] is None


def test_duplicate_ids_rejected():
    ms = [memory("a", NOW), memory("a", NOW)]
    with pytest.raises(ValueError): rank(ms, {"a": 1}, Q, "semantic")
    with pytest.raises(ValueError): evaluate(["a", "a"], {"relevance_grades": {"a": 1}}, 2)
