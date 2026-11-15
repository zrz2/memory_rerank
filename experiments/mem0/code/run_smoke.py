"""Bounded Mem0 2.2.1 smoke test: local BGE-M3, audited DeepSeek, conflict case."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MEM0 = ROOT / "experiments/mem0"
RUN = MEM0 / "runs/smoke"
RUN.mkdir(parents=True, exist_ok=True)
RESULTS = MEM0 / "results/smoke.json"
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
        "config": {"collection_name": "mem0_smoke", "embedding_model_dims": 1024,
                   "path": str(RUN / "qdrant"), "on_disk": True},
    },
    "llm": {"provider": "openai", "config": {"api_key": "unused"}},
    "history_db_path": str(RUN / "history.db"),
    "version": "v1.1",
}
memory = Memory.from_config(config)
memory.llm = AuditedDeepSeek({})
USER = "smoke_user"

turns = [
    "I moved to Beijing in March 2025 for a new job. I love it here.",
    "By the way, I am allergic to peanuts. Please remember that.",
    "Actually, the Beijing move was temporary. I relocated to Shanghai in June 2025 and I live there now.",
]
out = {"versions": {}, "turns": [], "searches": []}
from importlib.metadata import version
out["versions"] = {"mem0ai": version("mem0ai"), "qdrant-client": version("qdrant-client"),
                   "sentence-transformers": version("sentence-transformers"),
                   "embedding_model": MODEL}

for i, text in enumerate(turns, 1):
    result = memory.add([{"role": "user", "content": text}], user_id=USER)
    all_memories = memory.get_all(filters={"user_id": USER}, top_k=50)
    out["turns"].append({"turn": i, "input": text, "add_result": result,
                         "all_memories": all_memories})
    print(f"\n=== turn {i} add ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"=== turn {i} store ===")
    print(json.dumps(all_memories, ensure_ascii=False, indent=2))

for query in ["Where does the user live now?", "Which food should the user avoid?"]:
    result = memory.search(query, filters={"user_id": USER}, top_k=5)
    out["searches"].append({"query": query, "result": result})
    print(f"\n=== search: {query} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
print("\nsaved", RESULTS)
