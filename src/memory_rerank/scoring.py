"""Pure ranking functions. Labels/answers never enter this module."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return parsed


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Scores must be finite")
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def available(memories: list[dict], now: str) -> list[dict]:
    cutoff = timestamp(now)
    ids = [m["id"] for m in memories]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate memory IDs")
    return [m for m in memories if timestamp(m["created_at"]) <= cutoff]


def valid_at(memory: dict, target_time: str) -> bool:
    """Explicit oracle intervals only; this function does not infer validity."""
    target = timestamp(target_time)
    start, end = memory.get("valid_from"), memory.get("valid_to")
    return ((not start or timestamp(start) <= target)
            and (not end or target < timestamp(end)))


METHODS = {"semantic", "semantic_recency", "ga_paper_score",
           "ga_task_oracle", "ga_task_validity_oracle"}


def rank(memories: list[dict], similarities: dict[str, float], query: dict,
         method: str, decay: float = 0.995) -> list[dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0,1]")
    now = timestamp(query["query_time"])
    nodes = available(memories, query["query_time"])
    if method == "ga_task_validity_oracle":
        nodes = [m for m in nodes if valid_at(m, query.get("target_time", query["query_time"]))]
    if not nodes:
        return []
    ages = []
    for m in nodes:
        accessed = timestamp(m.get("last_accessed_at", m["created_at"]))
        if accessed < timestamp(m["created_at"]) or accessed > now:
            raise ValueError(f"Invalid last_accessed_at: {m['id']}")
        ages.append((now - accessed).total_seconds() / 3600)
    recency_raw = [decay ** h for h in ages]
    # Common scaling cancels under min-max; avoid all values underflowing to 0.
    recency = minmax([decay ** (h - min(ages)) for h in ages])
    relevance_raw = [float(similarities[m["id"]]) for m in nodes]
    relevance = minmax(relevance_raw)
    importance_raw = [float(m.get("importance", 5)) for m in nodes]
    if any(not math.isfinite(x) or not 1 <= x <= 10 for x in importance_raw):
        raise ValueError("importance must be finite in [1,10]")
    importance = minmax(importance_raw)
    task_tags = set(query.get("task_tags", []))
    task = [len(task_tags & set(m.get("task_tags", []))) / max(1, len(task_tags)) for m in nodes]
    rows = []
    for i, m in enumerate(nodes):
        if method == "semantic":
            score = relevance_raw[i]
        elif method == "semantic_recency":
            score = relevance[i] + recency[i]
        else:
            score = relevance[i] + recency[i] + importance[i]
            if method.startswith("ga_task"):
                score += task[i]
        rows.append({"memory_id": m["id"], "score": score,
                     "relevance_raw": relevance_raw[i], "relevance": relevance[i],
                     "recency_raw": recency_raw[i], "recency": recency[i],
                     "importance_raw": importance_raw[i], "importance": importance[i],
                     "task": task[i], "importance_source": m.get("importance_source", "missing_constant_5")})
    rows.sort(key=lambda x: (-x["score"], x["memory_id"]))
    return rows


def touch_selected(memories: list[dict], selected_ids: list[str], now: str) -> None:
    """Optional online state update; offline experiments deliberately do not call it."""
    cutoff = timestamp(now)
    selected = set(selected_ids)
    for m in memories:
        if m["id"] in selected:
            if timestamp(m["created_at"]) > cutoff:
                raise ValueError("Cannot access a future memory")
            m["last_accessed_at"] = now
