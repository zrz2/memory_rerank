"""Mem0 2.2.1 on project diagnostic tasks: minimal project-linked reproduction."""
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MEM0 = ROOT / "experiments/mem0"
RUN = MEM0 / "runs/project"
RUN.mkdir(parents=True, exist_ok=True)
RESULTS = MEM0 / "results/project.json"
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(ROOT / "src"))

from mem0 import Memory
from mem0_adapter import AuditedDeepSeek

MODEL = str((ROOT / ".cache/models/bge-m3").resolve())
config = {
    "embedder": {
        "provider": "huggingface",
        "config": {"model": MODEL, "embedding_dims": 1024,
                   "model_kwargs": {"device": "cpu", "local_files_only": True}},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "mem0_project", "embedding_model_dims": 1024,
                   "path": str(RUN / "qdrant"), "on_disk": True},
    },
    "llm": {"provider": "openai", "config": {"api_key": "unused"}},
    "history_db_path": str(RUN / "history.db"),
    "version": "v1.1",
}
memory = Memory.from_config(config)
memory.llm = AuditedDeepSeek({})

TASKS = {t["task_id"]: t for t in json.loads(
    (ROOT / "data/diagnostic/tasks_v0.json").read_text(encoding="utf8"))}
SELECTED = ["address_current", "address_history"]


def source_id(item):
    metadata = item.get("metadata") or {}
    return metadata.get("source_memory_id")


out = {"versions": {"mem0ai": version("mem0ai"), "qdrant-client": version("qdrant-client"),
                    "sentence-transformers": version("sentence-transformers"),
                    "embedding_model": MODEL},
       "ingest": [], "tasks": []}

for task_id in SELECTED:
    task = TASKS[task_id]
    for mem in sorted(task["memories"], key=lambda x: x["created_at"]):
        result = memory.add(
            [{"role": "user", "content": mem["content"]}],
            user_id=task_id,
            metadata={"source_memory_id": mem["id"], "created_at": mem["created_at"]},
        )
        out["ingest"].append({"task_id": task_id, "source_memory_id": mem["id"],
                              "input": mem["content"], "result": result})
        print(f"[ingest {task_id}] {mem['id']} -> {json.dumps(result, ensure_ascii=False)}")

    stored = memory.get_all(filters={"user_id": task_id}, top_k=50)
    task_out = {"task_id": task_id, "stored": stored, "queries": []}
    print(f"\n[{task_id}] stored {len(stored.get('results', []))} memories")

    for query in task["queries"]:
        found = memory.search(query["text"], filters={"user_id": task_id},
                              top_k=5, explain=True)
        ranked = [source_id(x) for x in found.get("results", [])]
        grades = query.get("relevance_grades", {})
        obsolete = set(query.get("obsolete_ids", []))
        gold_rank = next((i + 1 for i, sid in enumerate(ranked) if grades.get(sid, 0) > 0), None)
        task_out["queries"].append({
            "query": query,
            "result": found,
            "ranked_source_ids": ranked,
            "gold_rank": gold_rank,
            "mrr": (1.0 / gold_rank) if gold_rank else 0.0,
            "stale_ids_in_top5": [sid for sid in ranked if sid in obsolete],
        })
        print(f"[{task_id}] Q={query['text']!r} ranked={ranked} "
              f"gold_rank={gold_rank} stale={[s for s in ranked if s in obsolete]}")
    out["tasks"].append(task_out)

RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
print("\nsaved", RESULTS)
