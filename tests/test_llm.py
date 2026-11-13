import io
import json
from unittest.mock import patch
import pytest
from memory_rerank.llm import DeepSeekClient, read_key


def setup(root):
    (root / "configs").mkdir()
    (root / "configs/llm.json").write_text(json.dumps({"provider": "deepseek", "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY", "temperature": 0,
        "max_tokens": 512, "timeout_seconds": 90}))


def test_missing_key_never_calls_network(tmp_path, monkeypatch):
    setup(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with patch("urllib.request.urlopen") as request:
        with pytest.raises(RuntimeError, match="missing"):
            DeepSeekClient(tmp_path).chat([])
        request.assert_not_called()


def test_env_file_and_process_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text('# comment\nDEEPSEEK_API_KEY="test-not-real"\n')
    assert read_key(tmp_path) == "test-not-real"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "override-test")
    assert read_key(tmp_path) == "override-test"


def test_deepseek_endpoint_ignores_openai_environment(tmp_path, monkeypatch):
    setup(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example")
    reply = {"id": "sample", "model": "deepseek-chat", "usage": {"total_tokens": 3},
             "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
    with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as request:
        out = DeepSeekClient(tmp_path).chat([{"role": "user", "content": "OK"}], max_tokens=8)
        req = request.call_args.args[0]
        assert req.full_url == "https://api.deepseek.com/chat/completions"
        assert json.loads(req.data)["model"] == "deepseek-chat"
        assert "test-only" not in json.dumps(out)
        assert out["text"] == "OK" and out["usage"]["total_tokens"] == 3
