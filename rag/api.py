from __future__ import annotations

import time

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from rag.service import get_app_graph

app = FastAPI(title="RAG Knowledge Platform", version="1.0.0")

REQUESTS = Counter("rag_requests_total", "Questions answered", ["outcome"])
LATENCY = Histogram("rag_request_latency_seconds", "End-to-end latency",
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10))
GROUNDED = Histogram("rag_groundedness_score", "Groundedness of returned answers",
                     buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 1.0))


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: Ask):
    t = time.perf_counter()
    out = get_app_graph().invoke({"question": req.question})
    LATENCY.observe(time.perf_counter() - t)
    REQUESTS.labels("abstained" if out.get("abstained") else "answered").inc()
    GROUNDED.observe(out.get("grounded_score", 0))
    return {k: out.get(k) for k in ("answer", "sources", "grounded_score", "abstained")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
