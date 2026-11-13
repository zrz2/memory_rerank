"""Check local configuration; --live makes ONE short, billable DeepSeek request."""
from datetime import datetime, timezone
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memory_rerank.llm import DeepSeekClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Send one small DeepSeek request")
    parser.add_argument("--embedding", action="store_true", help="Load BGE-M3 and encode Chinese/English")
    args = parser.parse_args()
    client = DeepSeekClient(ROOT)
    status = {"checked_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "llm": client.config,
              "api_key_present": client.configured,
              "git_name": subprocess.check_output(["git", "config", "--local", "user.name"], cwd=ROOT, text=True).strip(),
              "git_email": subprocess.check_output(["git", "config", "--local", "user.email"], cwd=ROOT, text=True).strip(),
              "live_status": "not_requested", "embedding_status": "not_requested"}
    if args.embedding:
        import numpy as np
        from memory_rerank.embeddings import DenseEncoder
        encoder = DenseEncoder(ROOT)
        docs = encoder.encode(["用户现在的收货地址是上海。", "The current shipping address is Shanghai.",
                               "Jazz music is my favorite."], "passage")
        query = encoder.encode(["快递应该寄到哪里？"], "query")
        if (docs.shape != (3, 1024) or query.shape != (1, 1024)
                or not np.isfinite(docs).all() or not np.isfinite(query).all()
                or not np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-5)
                or not np.allclose(np.linalg.norm(query, axis=1), 1, atol=1e-5)):
            raise RuntimeError("BGE-M3 output validation failed")
        status["embedding_status"] = "passed"
        status["embedding"] = {"config": encoder.config, "shape": list(docs.shape),
                               "norms": np.linalg.norm(docs, axis=1).tolist(),
                               "similarities": (docs @ query[0]).tolist(), "stats": encoder.stats}
    if args.live:
        if not client.configured:
            status["live_status"] = "blocked_missing_key"
        else:
            try:
                reply = client.chat([{"role": "user", "content": "Reply with exactly OK."}], max_tokens=8)
                status["live_status"] = "passed" if reply["text"].strip() == "OK" else "response_received"
                status["live_response"] = reply
            except RuntimeError as exc:
                status["live_status"] = "failed"
                status["error"] = str(exc)
    destination = ROOT / "reports/environment_check.json"
    destination.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(status, ensure_ascii=True, indent=2))
    if status["live_status"] in {"failed", "blocked_missing_key"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
