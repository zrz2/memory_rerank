"""Six frozen source-order questions, all historical turns; local retrieval only."""
from __future__ import annotations
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memory_rerank.embeddings import DenseEncoder
from memory_rerank.metrics import evaluate
from memory_rerank.scoring import available, rank


def iso(value):
    cleaned = re.sub(r"\s*\([^)]*\)", "", value)
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc).isoformat()


def main():
    selected = json.loads((ROOT / "data/splits/longmemeval_smoke_ids.json").read_text(encoding="utf8"))
    source = ROOT / "data/raw/longmemeval/longmemeval_s_cleaned.json"
    ids = {x["question_id"] for x in selected}
    data = [x for x in json.loads(source.read_text(encoding="utf8")) if x["question_id"] in ids]
    encoder = DenseEncoder(ROOT)
    rows = []
    start = time.perf_counter()
    for item in data:
        qid = item["question_id"]
        query = {"query_id": qid, "text": item["question"], "query_time": iso(item["question_date"])}
        memories, grades = [], {}
        for sid, date, turns in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
            for turn_index, turn in enumerate(turns):
                mid = f"{sid}:{turn_index}"
                memories.append({"id": mid, "content": f"{turn['role']}: {turn['content']}",
                                 "created_at": iso(date), "last_accessed_at": iso(date),
                                 "importance": 5, "importance_source": "missing_constant_5"})
                if turn.get("has_answer"):
                    grades[mid] = 1
        visible = available(memories, query["query_time"])
        visible_ids = {m["id"] for m in visible}
        if set(grades) - visible_ids:
            raise ValueError(f"Gold evidence outside temporal visibility: {qid}")
        # Token windows cover complete turns; aggregate maximum chunk similarity per turn.
        chunks, owners = [], []
        for m in visible:
            tokens = encoder.model.tokenizer.encode(m["content"], add_special_tokens=False)
            for offset in range(0, max(1, len(tokens)), 160):
                piece = tokens[offset:offset+200]
                chunks.append(encoder.model.tokenizer.decode(piece, skip_special_tokens=True))
                owners.append(m["id"])
                if offset + 200 >= len(tokens):
                    break
        print(f"{qid}: {len(visible)} turns, {len(chunks)} chunks, {len(grades)} evidence turns", flush=True)
        vectors = encoder.encode(chunks, "passage")
        qvec = encoder.encode([query["text"]], "query")[0]
        sims = {m["id"]: -1.0 for m in visible}
        for mid, value in zip(owners, vectors @ qvec):
            sims[mid] = max(sims[mid], float(value))
        gold_query = {**query, "relevance_grades": grades}
        for method in ["semantic", "semantic_recency", "ga_paper_score"]:
            rank_start = time.perf_counter()
            ranked = rank(visible, sims, query, method)
            rank_seconds = time.perf_counter() - rank_start
            for k in [1, 3, 5, 10]:
                rows.append({"question_id": qid, "question_type": item["question_type"],
                             "method": method, "turns": len(visible), "chunks": len(chunks),
                             "filtered_future_turns": len(memories)-len(visible),
                             "ranking_seconds": rank_seconds, "metric": evaluate([x["memory_id"] for x in ranked], gold_query, k)})
            (ROOT / "runs" / f"lme_{qid}_{method}.json").write_text(json.dumps({
                "query": query, "gold_ids": sorted(grades), "ranked_scores": ranked}, indent=2), encoding="utf8")
        print(qid, "completed", flush=True)
    result = {"scope": "six_question_pipeline_smoke_not_official_benchmark", "date": "2026-11-13",
              "model": encoder.config, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "python": platform.python_version(), "elapsed_seconds": time.perf_counter()-start,
              "encoder_stats": encoder.stats, "rows": rows,
              "limitations": ["Importance is constant 5, not LLM scoring; GA ranking degenerates to semantic+recency.",
                              "last_accessed initialized to session creation; no real access log.",
                              "Timestamp timezone assumed UTC; only relative time used.",
                              "Token-window max pooling per turn; not original official retrieval pipeline.",
                              "No tuning or statistical claim; six frozen smoke samples."]}
    result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    (ROOT / "runs/longmemeval_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print("Saved runs/longmemeval_smoke.json", encoder.stats, flush=True)


if __name__ == "__main__":
    main()
