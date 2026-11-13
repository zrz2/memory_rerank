"""Download integrity checks using local mocked HTTP ranges (no network)."""
import hashlib
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('download_ranges', Path(__file__).resolve().parents[1] / 'scripts/download_longmemeval.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

class Response:
    status = 206
    def __init__(self, data, start, end, total):
        self.data = data
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}'}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return self.data

def mock_source(monkeypatch, content):
    starts = []
    def open_range(request, timeout):
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        starts.append(start)
        return Response(content[start:end+1], start, end, len(content))
    monkeypatch.setattr(download.urllib.request, 'urlopen', open_range)
    return starts

def test_resume_verified_prefix_and_final_hash(tmp_path, monkeypatch):
    content = b'abcdefgh' * 200000
    dest = tmp_path / 'model.bin'
    dest.with_suffix('.bin.part').write_bytes(content[:12345])
    starts = mock_source(monkeypatch, content)
    expected = hashlib.sha256(content).hexdigest()
    assert download.fetch('https://example.test/model', dest, len(content), expected) == expected
    assert min(starts) == 12345
    assert dest.read_bytes() == content
    assert not dest.with_suffix('.bin.part').exists()

def test_corrupt_prefix_never_promoted(tmp_path, monkeypatch):
    content = b'correct model bytes'
    dest = tmp_path / 'model.bin'
    dest.with_suffix('.bin.part').write_bytes(b'bad')
    mock_source(monkeypatch, content)
    with pytest.raises(ValueError, match='integrity'):
        download.fetch('https://example.test/model', dest, len(content), hashlib.sha256(content).hexdigest())
    assert not dest.exists()

def test_invalid_range_never_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(download.time, 'sleep', lambda _: None)
    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **k: Response(b'abc', 1, 3, 3))
    dest = tmp_path / 'model.bin'
    with pytest.raises(ValueError, match='Bad range'):
        download.fetch('https://example.test/model', dest, 3)
    assert not dest.exists()
