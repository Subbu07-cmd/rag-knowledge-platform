"""LangGraph RAG workflow: retrieve -> grade -> generate -> verify (or abstain)."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag.chunking import Chunk
from rag.llm import groundedness

ABSTAIN = "I don't know based on the available documents."


class RagState(TypedDict, total=False):
    question: str
    retrieved: list[tuple[Chunk, float]]
    relevant: list[Chunk]
    answer: str
    sources: list[str]
    grounded_score: float
    abstained: bool


def build_graph(store, embedder, generator, k: int = 4, min_score: float = 0.15, min_grounded: float = 0.6):
    def retrieve(state: RagState) -> RagState:
        qv = embedder.embed([state["question"]])[0]
        return {"retrieved": store.search(qv, k)}

    def grade(state: RagState) -> RagState:
        return {"relevant": [c for c, s in state["retrieved"] if s >= min_score]}

    def route_after_grade(state: RagState) -> str:
        return "generate" if state["relevant"] else "abstain"

    def generate(state: RagState) -> RagState:
        answer = generator.generate(state["question"], state["relevant"])
        return {"answer": answer, "sources": sorted({c.source for c in state["relevant"]})}

    def verify(state: RagState) -> RagState:
        if state["answer"].startswith("I don't know"):
            return {"grounded_score": 1.0, "abstained": True, "sources": []}
        score = groundedness(state["answer"], state["relevant"])
        if score < min_grounded:
            return {"answer": ABSTAIN, "grounded_score": score, "abstained": True, "sources": []}
        return {"grounded_score": score, "abstained": False}

    def abstain(state: RagState) -> RagState:
        return {"answer": ABSTAIN, "sources": [], "grounded_score": 1.0, "abstained": True}

    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("abstain", abstain)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "abstain": "abstain"})
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    g.add_edge("abstain", END)
    return g.compile()
