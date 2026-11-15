"""DeepSeek transport for the official Mem0 v0.1.94 source tree.

Keeps the upstream prompts and response handling; only replaces the LLM
transport and adds a persisted budget/cache ledger.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mem0.llms.base import LLMBase
from memory_rerank.llm import DeepSeekClient


class LegacyAuditedDeepSeek(LLMBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.client = DeepSeekClient(ROOT)
        self.folder = ROOT / "experiments/mem0/runs/api/paper_mechanism"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "ledger.json"
        self.ledger = json.loads(self.path.read_text(encoding="utf8")) if self.path.exists() else {"calls": []}
        self.limit = 1600
        self.max_calls = 12
        self.max_cny = 6.0
        self.call_log = []

    def generate_response(self, messages, response_format=None, tools=None, tool_choice=None, **kwargs):
        if tools or tool_choice:
            raise ValueError("Tool calls are not used by the v0.1.94 four-operation pipeline")
        identity = {
            "messages": messages,
            "response_format": response_format,
            "llm": self.client.config,
            "max_tokens": self.limit,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache = self.folder / f"{key}.json"
        if cache.exists():
            response = json.loads(cache.read_text(encoding="utf8"))["response"]
        else:
            if any(x["key"] == key for x in self.ledger["calls"]):
                raise RuntimeError("Unresolved previous request; no paid retry")
            reserve = ((len(json.dumps(messages, ensure_ascii=False).encode()) + 2048) * 10 + self.limit * 30) / 1e6
            spent = sum(x["reserved_cny"] for x in self.ledger["calls"])
            if len(self.ledger["calls"]) >= self.max_calls or spent + reserve > self.max_cny:
                raise RuntimeError("Paper-mechanism run exceeded its call or budget cap")
            entry = {"key": key, "reserved_cny": reserve, "status": "reserved"}
            self.ledger["calls"].append(entry)
            self._save()
            response = self.client.chat(messages, max_tokens=self.limit, response_format=response_format)
            self.folder.joinpath(f"{key}.json").write_text(
                json.dumps({**identity, "response": response}, ensure_ascii=False, indent=2), encoding="utf8"
            )
            entry.update(
                status="response_saved",
                usage=response.get("usage"),
                returned_model=response.get("returned_model"),
                configured_model=response.get("configured_model"),
                elapsed_seconds=response.get("elapsed_seconds"),
            )
            self._save()
        if response.get("finish_reason") != "stop":
            raise ValueError("Incomplete Mem0 LLM output")
        self.call_log.append(
            {
                "response_text": response["text"],
                "usage": response.get("usage"),
                "returned_model": response.get("returned_model"),
            }
        )
        return response["text"]

    def _save(self):
        self.path.write_text(json.dumps(self.ledger, ensure_ascii=False, indent=2), encoding="utf8")
