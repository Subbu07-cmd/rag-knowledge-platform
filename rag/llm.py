"""Answer generation: Bedrock Claude in production, extractive fallback offline."""
from __future__ import annotations

import os
import re

from rag.chunking import Chunk
from rag.text import tokenize

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. Cite sources as [n]. "
    "If the context does not contain the answer, reply exactly: I don't know based on the available documents."
)


def _tokens(s: str) -> set[str]:
    return set(tokenize(s))


class ExtractiveGenerator:
    """Picks the sentences that best overlap the question. Never invents text."""

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        q = _tokens(question)
        scored = []
        for n, c in enumerate(chunks, 1):
            for sent in re.split(r"(?<=[.!?])\s+|\n+", c.text):
                overlap = len(q & _tokens(sent))
                if overlap:
                    scored.append((overlap, sent.strip(" -#*"), n))
        if not scored:
            return "I don't know based on the available documents."
        scored.sort(key=lambda x: -x[0])
        return " ".join(f"{s} [{n}]" for _, s, n in scored[:2])


class BedrockGenerator:
    def __init__(self) -> None:
        from langchain_aws import ChatBedrockConverse

        self.llm = ChatBedrockConverse(
            model=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            temperature=0,
            max_tokens=600,
        )

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, c in enumerate(chunks, 1))
        msgs = [("system", SYSTEM_PROMPT), ("human", f"Context:\n{context}\n\nQuestion: {question}")]
        return self.llm.invoke(msgs).content


def get_generator():
    if os.getenv("LLM_PROVIDER", "extractive").lower() == "bedrock":
        return BedrockGenerator()
    return ExtractiveGenerator()


def groundedness(answer: str, chunks: list[Chunk]) -> float:
    """Share of answer content words that appear in the retrieved context (0..1)."""
    a = _tokens(re.sub(r"\[\d+\]", "", answer))
    if not a:
        return 0.0
    ctx = set().union(*(_tokens(c.text) for c in chunks)) if chunks else set()
    return len(a & ctx) / len(a)
