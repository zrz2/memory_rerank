"""Explicit DeepSeek client. Does not inherit unrelated OPENAI_* settings."""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_key(root: Path, variable: str = "DEEPSEEK_API_KEY") -> str:
    value = os.environ.get(variable, "").strip()
    if value:
        return value
    path = root / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, separator, candidate = line.strip().partition("=")
            if separator and name.strip() == variable:
                return candidate.strip().strip('"').strip("'")
    return ""


class DeepSeekClient:
    def __init__(self, root: Path):
        self.root = root
        self.config = json.loads((root / "configs/llm.json").read_text(encoding="utf8"))
        if self.config["provider"] != "deepseek" or self.config["base_url"] != "https://api.deepseek.com":
            raise ValueError("Expected explicit official DeepSeek endpoint")

    @property
    def configured(self) -> bool:
        return bool(read_key(self.root, self.config["api_key_env"]))

    def chat(self, messages: list[dict], *, max_tokens: int | None = None) -> dict:
        key = read_key(self.root, self.config["api_key_env"])
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing. Set it in the project .env or process environment.")
        payload = {"model": self.config["model"], "messages": messages,
                   "temperature": self.config["temperature"], "stream": False,
                   "max_tokens": max_tokens or self.config["max_tokens"]}
        request = urllib.request.Request(self.config["base_url"] + "/chat/completions",
                                         data=json.dumps(payload).encode("utf8"),
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                                         method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not log raw response/request headers, which may contain credentials.
            raise RuntimeError(f"DeepSeek returned HTTP {exc.code}; check account, model access and balance.") from None
        except urllib.error.URLError:
            raise RuntimeError("DeepSeek network request failed; check connectivity.") from None
        return {"text": result["choices"][0]["message"].get("content", ""),
                "response_id": result.get("id"), "returned_model": result.get("model"),
                "usage": result.get("usage", {}), "finish_reason": result["choices"][0].get("finish_reason"),
                "elapsed_seconds": time.perf_counter()-start, "configured_model": self.config["model"]}
