"""Wires the store, embedder and generator together."""
from __future__ import annotations

import os
from functools import lru_cache

from rag.embeddings import get_embedder
from rag.graph import build_graph
from rag.llm import get_generator
from rag.store import VectorStore


@lru_cache(maxsize=1)
def get_app_graph(index_path: str | None = None):
    store = VectorStore.load(index_path or os.getenv("RAG_INDEX", ".index"))
    return build_graph(store, get_embedder(), get_generator())
