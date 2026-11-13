"""Inspect actual downloaded files. No labels are repaired or invented."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit():
    raw = ROOT / "data/raw"
    lme_path = raw / "longmemeval/longmemeval_s_cleaned.json"
    lme = load(lme_path)
    lme_summary = {
        "instances": len(lme), "categories": dict(Counter(x["question_type"] for x in lme)),
        "abstention": sum(x["question_id"].endswith("_abs") for x in lme),
        "alignment_failures": [], "missing_evidence_sessions": [],
        "answerable_without_marked_turns": [],
        "sha256": hashlib.sha256(lme_path.read_bytes()).hexdigest(),
        "file_bytes": lme_path.stat().st_size,
    }
    for x in lme:
        if not len(x["haystack_sessions"]) == len(x["haystack_dates"]) == len(x["haystack_session_ids"]):
            lme_summary["alignment_failures"].append(x["question_id"])
        if set(x["answer_session_ids"]) - set(x["haystack_session_ids"]):
            lme_summary["missing_evidence_sessions"].append(x["question_id"])
        if not x["question_id"].endswith("_abs") and not any(t.get("has_answer") for s in x["haystack_sessions"] for t in s):
            lme_summary["answerable_without_marked_turns"].append(x["question_id"])
    locomo_path = raw / "locomo/data/locomo10.json"
    locomo = load(locomo_path)
    loc_summary = {"conversations": len(locomo), "questions": sum(len(x["qa"]) for x in locomo),
                   "categories": dict(Counter(str(q["category"]) for x in locomo for q in x["qa"])),
                   "missing_evidence_questions": 0, "unresolved_evidence": [],
                   "turns": 0, "image_turns": 0, "duplicate_ids": [],
                   "sha256": hashlib.sha256(locomo_path.read_bytes()).hexdigest()}
    for x in locomo:
        turns = [t for k, v in x["conversation"].items() if k.startswith("session_") and isinstance(v, list) for t in v]
        ids = [t["dia_id"] for t in turns]
        if len(set(ids)) != len(ids): loc_summary["duplicate_ids"].append(x["sample_id"])
        loc_summary["turns"] += len(turns)
        loc_summary["image_turns"] += sum(bool(t.get("img_url")) for t in turns)
        for i, q in enumerate(x["qa"]):
            ev = q.get("evidence", [])
            if not ev: loc_summary["missing_evidence_questions"] += 1
            missing = [e for e in ev if e not in set(ids)]
            if missing: loc_summary["unresolved_evidence"].append({"sample_id": x["sample_id"], "question_index": i, "ids": missing})
    result = {"checked_on": "2026-11-13", "longmemeval_s_cleaned": lme_summary, "locomo10": loc_summary}
    (ROOT / "reports/dataset_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    audit()
