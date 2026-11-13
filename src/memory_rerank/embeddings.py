"""Pinned local multilingual E5 inference with content-addressed vector caching."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np


class DenseEncoder:
    def __init__(self, root: Path):
        self.root = root
        self.config = json.loads((root / "configs/embedding_model.json").read_text(encoding="utf8"))
        os.environ["HF_HOME"] = str(root / ".cache/huggingface")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.manual_seed(0)
        self.model = SentenceTransformer(str(root / ".cache/models/multilingual-e5-small"),
                                         device="cpu", local_files_only=True)
        self.model.max_seq_length = self.config["max_seq_length"]
        self.cache = root / ".cache/embeddings"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stats = {"texts": 0, "cache_hits": 0, "computed": 0, "truncated_texts": 0,
                      "encode_seconds": 0.0, "external_api_calls": 0}

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        if kind not in {"query", "passage"}:
            raise ValueError(kind)
        inputs = [self.config[f"{kind}_prefix"] + t for t in texts]
        paths = [self.cache / (hashlib.sha256((json.dumps(self.config, sort_keys=True) + t).encode()).hexdigest() + ".npy") for t in inputs]
        vectors = [None] * len(inputs)
        missing = []
        for i, path in enumerate(paths):
            if path.exists():
                vectors[i] = np.load(path, allow_pickle=False)
                self.stats["cache_hits"] += 1
            else:
                missing.append(i)
        lengths = self.model.tokenizer(inputs, truncation=False, add_special_tokens=True)["input_ids"] if inputs else []
        self.stats["truncated_texts"] += sum(len(x) > self.config["max_seq_length"] for x in lengths)
        if missing:
            start = time.perf_counter()
            batch = self.model.encode([inputs[i] for i in missing], batch_size=self.config["batch_size"],
                                      normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            self.stats["encode_seconds"] += time.perf_counter() - start
            for i, vector in zip(missing, batch):
                vectors[i] = vector
                np.save(paths[i], vector, allow_pickle=False)
        self.stats["computed"] += len(missing)
        self.stats["texts"] += len(inputs)
        return np.stack(vectors) if vectors else np.empty((0, self.model.get_sentence_embedding_dimension()))
