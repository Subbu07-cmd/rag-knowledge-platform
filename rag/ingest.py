"""Build a FAISS index from a folder or S3 prefix.

python -m rag.ingest --source data/docs --index .index
"""
from __future__ import annotations

import argparse
import time

from rag.chunking import chunk_documents
from rag.embeddings import get_embedder
from rag.loaders import load
from rag.store import VectorStore


def build_index(source: str, chunk_size: int = 500, overlap: int = 80, batch: int = 64) -> VectorStore:
    docs = load(source)
    chunks = chunk_documents(docs, chunk_size, overlap)
    embedder = get_embedder()
    store = None
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vecs = embedder.embed([c.text for c in part])
        store = store or VectorStore(vecs.shape[1])
        store.add(part, vecs)
    if store is None:
        raise SystemExit(f"No documents found in {source}")
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--index", default=".index")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    a = ap.parse_args()
    t = time.perf_counter()
    store = build_index(a.source, a.chunk_size, a.overlap)
    store.save(a.index)
    print(f"Indexed {len(store.chunks)} chunks in {time.perf_counter() - t:.2f}s -> {a.index}")


if __name__ == "__main__":
    main()
