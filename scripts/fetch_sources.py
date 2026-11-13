"""Restore pinned source metadata and LoCoMo; then run download_longmemeval.py."""
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
locks = json.loads((root / "configs/dataset_sources.json").read_text(encoding="utf8"))
manifest = []
for key, config in locks.items():
    folder = root / "data/raw" / key
    folder.mkdir(parents=True, exist_ok=True)
    names = (["README.md", "LICENSE", "src/retrieval/eval_utils.py", "src/evaluation/print_retrieval_metrics.py"]
             if key == "longmemeval" else ["README.MD", "LICENSE.txt", "data/locomo10.json", "task_eval/evaluation.py"])
    (folder / "repository.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    for name in names:
        url = f"https://raw.githubusercontent.com/{config['repository']}/{config['commit']}/{name}"
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(url, timeout=45).read()
        if name.endswith(".json"):
            json.loads(data)
        target.write_bytes(data)
        manifest.append({"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if key == "longmemeval":
        revision = config["dataset_revision"]
        (folder / "hf_metadata.json").write_text(json.dumps({"sha": revision}), encoding="utf8")
        url = f"https://huggingface.co/datasets/{config['dataset_repository']}/resolve/{revision}/README.md"
        (folder / "README_dataset.md").write_bytes(urllib.request.urlopen(url, timeout=45).read())
(root / "data/source_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
print("Pinned source files fetched. Run download_longmemeval.py --include-s next.")
