"""Recursive character splitter with overlap (no external dependencies)."""
from __future__ import annotations

from dataclasses import dataclass

from rag.loaders import Document

SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out, buf = [], ""
    for i, part in enumerate(parts):
        piece = (sep if i else "") + part
        if len(buf) + len(piece) <= size:
            buf += piece
        else:
            if buf:
                out.append(buf)
            buf = piece if len(piece) <= size else ""
            if len(piece) > size:
                out.extend(_split(piece, size, rest))
    if buf:
        out.append(buf)
    return out


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = _split(doc.text, size, SEPARATORS)
        for i, piece in enumerate(pieces):
            prefix = pieces[i - 1][-overlap:] if i and overlap else ""
            text = (prefix + piece).strip()
            if text:
                chunks.append(Chunk(text, doc.source, len(chunks)))
    return chunks
