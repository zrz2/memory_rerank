"""Run the four-operation pipeline from the official Mem0 v0.1.94 source tree.

This is a mechanism demo, not a LOCOMO benchmark: one controlled conversation
is used to expose ADD / UPDATE / DELETE / NONE decisions.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPO = HERE / "vendor" / "mem0_repo"
RUN = ROOT / "experiments/mem0/runs/paper_mechanism"
OUT = ROOT / "experiments/mem0/results/paper_mechanism.json"
USER_ID = "paper_mechanism_user"

os.environ["MEM0_DIR"] = str(RUN / "mem0_home")
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if not (REPO / "mem0/memory/main.py").exists():
    raise SystemExit(
        "Official Mem0 v0.1.94 source is missing. Run:\n"
        "git clone --depth 1 --branch v0.1.94 --filter=blob:none --sparse "
        "git@github.com:mem0ai/mem0.git experiments/mem0/paper_mechanism/vendor/mem0_repo\n"
        "git -C experiments/mem0/paper_mechanism/vendor/mem0_repo sparse-checkout set mem0"
    )
sys.path.insert(0, str(REPO))

import mem0  # noqa: E402
from mem0 import Memory  # noqa: E402

if Path(mem0.__file__).resolve().parents[1] != REPO.resolve():
    raise SystemExit(f"Imported mem0 from {mem0.__file__}, not the official v0.1.94 source tree")

from legacy_adapter import LegacyAuditedDeepSeek  # noqa: E402
from mem0.configs.base import MemoryConfig  # noqa: E402

TURNS = [
    "I moved to Beijing in March 2025 and work at ByteDance.",
    "By the way, I am allergic to peanuts.",
    "Actually, I relocated to Shanghai in June 2025; I live there now.",
    "Correction: I never joined ByteDance.",
    "I am still allergic to peanuts.",
]
QUERIES = {
    "current_home": "Where does the user live now?",
    "history_home": "Where did the user live before?",
    "job": "Where does the user work?",
    "allergy": "Which food should the user avoid?",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_memory():
    config = {
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": str(ROOT / ".cache/models/bge-m3"),
                "embedding_dims": 1024,
                "model_kwargs": {"device": "cpu", "local_files_only": True},
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_paper_094",
                "embedding_model_dims": 1024,
                "path": str(RUN / "qdrant"),
                "on_disk": True,
            },
        },
        "llm": {"provider": "openai", "config": {"api_key": "unused"}},
        "history_db_path": str(RUN / "history.db"),
        "version": "v1.1",
    }
    memory = Memory(MemoryConfig(**config))
    memory.llm = LegacyAuditedDeepSeek({})
    return memory


def list_memories(memory):
    result = memory.get_all(user_id=USER_ID)
    return [
        {
            "id": item["id"],
            "memory": item["memory"],
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        }
        for item in result.get("results", [])
    ]


def main():
    for name in ("qdrant", "history.db", "mem0_home"):
        target = RUN / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    memory = build_memory()
    records = []
    for index, text in enumerate(TURNS, 1):
        memory.llm.call_log = []
        result = memory.add(text, user_id=USER_ID, metadata={"source_memory_id": f"turn_{index}"})
        records.append(
            {
                "turn": index,
                "input": text,
                "decisions": result.get("results", []),
                "llm_responses": list(memory.llm.call_log),
                "memories_after": list_memories(memory),
            }
        )

    queries = {}
    for name, query in QUERIES.items():
        result = memory.search(query, user_id=USER_ID, limit=5)
        queries[name] = {
            "query": query,
            "results": [
                {
                    "id": item.get("id"),
                    "memory": item.get("memory"),
                    "score": item.get("score"),
                    "created_at": item.get("created_at"),
                }
                for item in result.get("results", [])
            ],
        }

    ledger_path = ROOT / "experiments/mem0/runs/api/paper_mechanism/ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf8")) if ledger_path.exists() else {"calls": []}
    calls = [
        {
            "status": item.get("status"),
            "reserved_cny": item.get("reserved_cny"),
            "usage": item.get("usage"),
            "configured_model": item.get("configured_model"),
            "returned_model": item.get("returned_model"),
            "elapsed_seconds": item.get("elapsed_seconds"),
        }
        for item in ledger["calls"]
    ]

    output = {
        "provenance": {
            "paper": "arXiv:2504.19413v1 (2025-04-28); paper text: Code can be found at https://mem0.ai/research",
            "official_repo": "https://github.com/mem0ai/mem0",
            "tag": git("describe", "--tags", "--exact-match"),
            "commit": git("rev-parse", "HEAD"),
            "commit_date": git("show", "-s", "--format=%ci", "HEAD"),
            "source_path": relative(REPO),
            "loaded_from": relative(mem0.__file__),
            "source_note": "mem0/memory/main.py calls get_update_memory_messages and handles ADD/UPDATE/DELETE/NONE; retrieval limit is 5.",
        },
        "run": {
            "user_id": USER_ID,
            "embedder": "BAAI/bge-m3, 1024-d, CPU, local_files_only",
            "vector_store": "Qdrant local on_disk",
            "llm_config": "DeepSeek deepseek-chat via project adapter",
        },
        "conversation": records,
        "queries": queries,
        "llm_calls": calls,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"output": str(OUT), "turns": len(records), "llm_calls": len(calls)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
