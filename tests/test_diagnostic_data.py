import json
from pathlib import Path
from memory_rerank.scoring import available, timestamp


def test_diagnostic_integrity_and_evidence_visibility():
    path = Path(__file__).resolve().parents[1] / "data/diagnostic/tasks_v0.json"
    tasks = json.loads(path.read_text(encoding="utf8"))
    assert len(tasks) == 10
    assert len({t["task_id"] for t in tasks}) == 10
    queries = set()
    for task in tasks:
        for q in task["queries"]:
            assert q["query_id"] not in queries
            queries.add(q["query_id"])
            visible = {m["id"] for m in available(task["memories"], q["query_time"])}
            assert set(q["relevance_grades"]) <= visible
            assert set(q["obsolete_ids"]) <= visible
            for group in q["required_evidence_sets"]:
                assert group and set(group) <= visible
                assert all(q["relevance_grades"][mid] > 0 for mid in group)
        for m in task["memories"]:
            if m.get("valid_from") and m.get("valid_to"):
                assert timestamp(m["valid_from"]) < timestamp(m["valid_to"])
