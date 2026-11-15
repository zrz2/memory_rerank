"""One-shot Mem0 pipeline demo: extraction + update (four operations) + retrieval."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MEM0 = ROOT / "experiments/mem0"
RUN = MEM0 / "runs/demo"
OLD_CACHE = MEM0 / "vendor/mem0_059"
OLD_WHEEL = MEM0 / "vendor/mem0ai-0.1.59-py3-none-any.whl"
MODEL = str((ROOT / ".cache/models/bge-m3").resolve())

os.environ["MEM0_TELEMETRY"] = "False"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=["old", "new"], required=True)
parser.add_argument("--out", default=None)
args = parser.parse_args()

if args.engine == "old" and not (OLD_CACHE / "mem0" / "__init__.py").exists():
    if not OLD_WHEEL.exists():
        raise SystemExit("run: pip download mem0ai==0.1.59 --no-deps -d experiments/mem0/vendor")
    OLD_CACHE.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(OLD_WHEEL).extractall(OLD_CACHE)

if args.engine == "old":
    sys.path.insert(0, str(OLD_CACHE))
sys.path.insert(0, str(ROOT / "src"))

from mem0 import Memory
from mem0.llms.base import LLMBase
from memory_rerank.llm import DeepSeekClient

USER = f"demo_{args.engine}"
TURNS = [
    "我住在北京。",
    "我搬到上海了。",
    "我现在住上海浦东。",
    "我住浦东。",
    "我现在住在上海浦东。",
    "我之前在字节跳动工作。",
    "其实我从来没在字节跳动工作过，那是我记错了。",
]
QUERIES = ["我现在住哪？", "我以前住哪？"]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


class LedgerLLM(LLMBase):
    def __init__(self, folder, cap_calls=30, cap_cny=12.0):
        super().__init__(None)
        self.client = DeepSeekClient(ROOT)
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "ledger.json"
        self.ledger = json.loads(self.path.read_text(encoding="utf8")) if self.path.exists() else {"calls": []}
        self.cap_calls = cap_calls
        self.cap_cny = cap_cny
        self.responses = []

    def generate_response(self, messages, response_format=None, tools=None, tool_choice=None, **kwargs):
        if tools or tool_choice:
            raise ValueError("tool calls are not used by this demo")
        limit = 1200
        raw = json.dumps(messages, ensure_ascii=False)
        key = hashlib.sha256((raw + json.dumps(response_format, sort_keys=True) + str(limit)).encode()).hexdigest()
        cache = self.folder / f"{key}.json"
        if cache.exists():
            text = json.loads(cache.read_text(encoding="utf8"))["text"]
            self.responses.append(text)
            return text
        reserve = ((len(raw.encode()) + 2048) * 10 + limit * 30) / 1e6
        spent = sum(c["reserved_cny"] for c in self.ledger["calls"])
        if spent + reserve > self.cap_cny or len(self.ledger["calls"]) >= self.cap_calls:
            raise RuntimeError("demo budget or call cap exceeded")
        entry = {"key": key, "reserved_cny": reserve, "status": "reserved"}
        self.ledger["calls"].append(entry)
        save(self.path, self.ledger)
        response = self.client.chat(messages, max_tokens=limit, response_format=response_format)
        save(cache, {"messages": messages, "text": response["text"]})
        entry.update(status="response_saved", usage=response.get("usage"),
                     returned_model=response.get("returned_model"))
        save(self.path, self.ledger)
        if response.get("finish_reason") != "stop":
            raise ValueError("incomplete LLM output")
        self.responses.append(response["text"])
        return response["text"]


def normalize(value):
    if isinstance(value, dict):
        return value.get("results", value)
    return value


def event_rows(result):
    rows = []
    for item in normalize(result) or []:
        if isinstance(item, dict):
            rows.append({k: item.get(k) for k in ("id", "memory", "event", "previous_memory", "old_memory")})
    return rows


def store_rows(memory):
    if args.engine == "old":
        data = memory.get_all(user_id=USER, limit=50)
    else:
        data = memory.get_all(filters={"user_id": USER}, top_k=50)
    return normalize(data)


def search(memory, query):
    if args.engine == "old":
        data = memory.search(query, user_id=USER, limit=5)
    else:
        data = memory.search(query, filters={"user_id": USER}, top_k=5)
    rows = []
    for item in normalize(data) or []:
        if isinstance(item, dict):
            rows.append({"id": item.get("id"), "memory": item.get("memory") or item.get("data"),
                         "score": item.get("score")})
    return rows


def main():
    run_dir = RUN / f"{args.engine}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    config = {
        "embedder": {"provider": "huggingface",
                     "config": {"model": MODEL, "embedding_dims": 1024,
                                "model_kwargs": {"device": "cpu", "local_files_only": True}}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": f"demo_{args.engine}",
                                    "embedding_model_dims": 1024,
                                    "path": str(run_dir / "qdrant"), "on_disk": False}},
        "llm": {"provider": "deepseek",
                "config": {"model": "deepseek-chat", "api_key": "unused",
                           "deepseek_base_url": "https://api.deepseek.com"}},
        "history_db_path": str(run_dir / "history.db"),
        "version": "v1.1",
    }
    memory = Memory.from_config(config)
    memory.llm = LedgerLLM(MEM0 / "runs/api/demo")

    out = {
        "engine": args.engine,
        "mem0_file": str(Path(sys.modules["mem0"].__file__)),
        "mem0_package_version": getattr(sys.modules["mem0"], "__version__", None),
        "mem0_metadata_version": version("mem0ai"),
        "qdrant_client": version("qdrant-client"),
        "sentence_transformers": version("sentence-transformers"),
        "embedding_model": MODEL,
        "turns": [],
    }
    for index, text in enumerate(TURNS, 1):
        result = memory.add([{"role": "user", "content": text}], user_id=USER)
        last_output = None
        if getattr(memory.llm, "responses", None):
            try:
                last_output = json.loads(memory.llm.responses[-1])
            except json.JSONDecodeError:
                last_output = memory.llm.responses[-1][:500]
        out["turns"].append({"turn": index, "input": text, "events": event_rows(result),
                             "raw": result, "last_llm_output": last_output,
                             "store": store_rows(memory)})
        print(f"\n=== turn {index}: {text}")
        print(json.dumps(out["turns"][-1]["events"], ensure_ascii=False, indent=2))

    out["searches"] = []
    for query in QUERIES:
        rows = search(memory, query)
        out["searches"].append({"query": query, "results": rows})
        print(f"\n=== search: {query}")
        print(json.dumps(rows, ensure_ascii=False, indent=2))

    out["ledger"] = json.loads((MEM0 / "runs/api/demo/ledger.json").read_text(encoding="utf8"))
    target = Path(args.out) if args.out else MEM0 / "results/demo" / f"{args.engine}.json"
    save(target, out)
    print(f"\nsaved {target}")


if __name__ == "__main__":
    main()
