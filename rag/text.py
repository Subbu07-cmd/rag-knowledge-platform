"""Shared lightweight tokenizer used by the offline embedder and groundedness check."""
from __future__ import annotations

import re

STOPWORDS = frozenset([
    "the", "and", "for", "with", "that", "this", "from", "into", "what", "when", "where", "which", "who",
    "how", "why", "are", "was", "were", "can", "does", "did", "should", "would", "could", "must", "not",
    "but", "you", "your", "our", "their", "its", "have", "has", "had", "then", "than", "also", "only",
    "any", "all", "is", "do", "be",
])


def _stem(t: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(t) > 4 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in STOPWORDS]
