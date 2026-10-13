"""FAISS vector store (inner product on normalized vectors = cosine similarity)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag.chunking import Chunk

try:
    import faiss
except ImportError:  # pragma: no cover - lets unit tests run without the native wheel
    faiss = None


class _NumpyIndex:
    def __init__(self, dim: int) -> None:
        self.m = np.zeros((0, dim), dtype="float32")

    def add(self, v: np.ndarray) -> None:
        self.m = np.vstack([self.m, v])

    def search(self, q: np.ndarray, k: int):
        scores = q @ self.m.T
        idx = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, idx, axis=1), idx


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) if faiss else _NumpyIndex(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        k = min(k, len(self.chunks))
        if k == 0:
            return []
        scores, idx = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0], strict=False) if i >= 0]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if faiss:
            faiss.write_index(self.index, str(p / "index.faiss"))
        else:
            np.save(p / "index.npy", self.index.m)
        (p / "chunks.json").write_text(json.dumps({"dim": self.dim, "chunks": [asdict(c) for c in self.chunks]}))

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        p = Path(path)
        meta = json.loads((p / "chunks.json").read_text())
        store = cls(meta["dim"])
        if faiss and (p / "index.faiss").exists():
            store.index = faiss.read_index(str(p / "index.faiss"))
        else:
            store.index.add(np.load(p / "index.npy"))
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        return store
