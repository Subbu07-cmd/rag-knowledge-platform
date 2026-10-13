"""Evaluate the RAG pipeline: accuracy, retrieval hit rate, abstention, latency and cost.

python -m eval.run_eval --index .index --dataset eval/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from rag.service import get_app_graph

# Approximate Bedrock on-demand prices (USD per 1K tokens); update for your model/region.
PRICE_IN, PRICE_OUT = 0.003, 0.015


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def evaluate(index: str, dataset: str) -> dict:
    graph = get_app_graph(index)
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    latencies, results = [], []
    for row in rows:
        t = time.perf_counter()
        out = graph.invoke({"question": row["question"]})
        latencies.append(time.perf_counter() - t)
        answer = out["answer"].lower()
        correct = all(e.lower() in answer for e in row["expected"])
        hit = row["source"] is None or any(s.endswith(row["source"]) for s in out.get("sources") or [])
        ctx = " ".join(c.text for c, _ in out.get("retrieved", []))
        cost = approx_tokens(ctx + row["question"]) / 1000 * PRICE_IN + approx_tokens(out["answer"]) / 1000 * PRICE_OUT
        results.append({"question": row["question"], "correct": correct, "retrieval_hit": hit,
                        "abstained": out.get("abstained"), "grounded": out.get("grounded_score"), "cost_usd": cost})
    lat = sorted(latencies)
    return {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in results) / len(rows),
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in results) / len(rows),
        "abstention_rate": sum(bool(r["abstained"]) for r in results) / len(rows),
        "latency_p50_ms": round(statistics.median(lat) * 1000, 1),
        "latency_p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1000, 1),
        "est_cost_per_query_usd": round(sum(r["cost_usd"] for r in results) / len(rows), 5),
        "details": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=".index")
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    a = ap.parse_args()
    report = evaluate(a.index, a.dataset)
    summary = {k: v for k, v in report.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, indent=2))
    if report["accuracy"] < a.min_accuracy:
        raise SystemExit(f"Accuracy {report['accuracy']:.2f} below threshold {a.min_accuracy}")


if __name__ == "__main__":
    main()
