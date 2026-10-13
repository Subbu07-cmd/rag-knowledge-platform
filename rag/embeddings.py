"""Embedding providers. All return L2-normalized float32 vectors."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from rag.text import tokenize


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype("float32")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for offline use and tests."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype="float32")
        for row, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
                m[row, h % self.dim] += 1.0
        return _normalize(m)


class BedrockEmbedder:
    """Amazon Titan Text Embeddings v2 via Bedrock."""

    def __init__(self, model_id: str | None = None, dim: int = 1024) -> None:
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.model_id = model_id or os.getenv("EMBEDDINGS_MODEL_ID", "amazon.titan-embed-text-v2:0")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": t[:8000], "dimensions": self.dim, "normalize": True}),
            )
            vecs.append(json.loads(resp["body"].read())["embedding"])
        return _normalize(np.array(vecs, dtype="float32"))


def get_embedder():
    if os.getenv("EMBEDDINGS_PROVIDER", "hashing").lower() == "bedrock":
        return BedrockEmbedder()
    return HashingEmbedder()
