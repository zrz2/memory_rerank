"""Local diagnostic metrics, not an implementation of any official QA judge."""
from __future__ import annotations
import math


def evaluate(ranked_ids: list[str], query: dict, k: int) -> dict:
    if k < 1 or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("K must be positive; ranked IDs must be unique")
    grades = query.get("relevance_grades", {})
    if any(not math.isfinite(float(g)) or float(g) < 0 for g in grades.values()):
        raise ValueError("Invalid relevance grade")
    gold = {mid for mid, grade in grades.items() if grade > 0}
    top = ranked_ids[:k]
    obsolete = set(query.get("obsolete_ids", []))
    result = {"k": k, "returned": len(top), "gold_count": len(gold),
              "obsolete_count": len(set(top) & obsolete),
              "obsolete_rate": len(set(top) & obsolete) / len(top) if top else None}
    if not gold:
        result.update(recall=None, mrr=None, ndcg=None, complete_evidence=None)
        # Absence of evidence is not a calibrated abstention detector.
        return result
    result["recall"] = len(set(top) & gold) / len(gold)
    result["mrr"] = next((1 / (i + 1) for i, mid in enumerate(top) if mid in gold), 0.0)
    dcg = sum((2 ** grades.get(mid, 0) - 1) / math.log2(i + 2) for i, mid in enumerate(top))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    result["ndcg"] = dcg / idcg
    groups = query.get("required_evidence_sets", [sorted(gold)])
    result["complete_evidence"] = float(any(set(g).issubset(top) for g in groups if g))
    return result
