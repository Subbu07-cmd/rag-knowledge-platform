# RAG-Powered AI Knowledge Platform

> **About this repo:** my production work lives in private enterprise repositories, so this is a public reference build of the same patterns, written from scratch with sample data and placeholder account IDs.

An end-to-end **retrieval-augmented generation (RAG)** system. It covers document ingestion and chunking, an **embeddings pipeline**, **FAISS** vector search, a **LangGraph** agentic answer workflow with retrieval grading and groundedness checks, **AWS Bedrock (Anthropic Claude)** for generation, a **FastAPI** service with **Prometheus** metrics, and an **evaluation framework** that tracks accuracy, latency and cost.

```
 S3 / local docs ─▶ loader ─▶ chunker ─▶ embedder (Bedrock Titan) ─▶ FAISS index
                                                                          │
 question ─▶ ┌─────────┐   ┌────────────┐   ┌───────────┐   ┌────────────────────┐
             │retrieve │──▶│ grade docs │──▶│ generate  │──▶│ groundedness check │──▶ answer + sources
             └─────────┘   └────────────┘   │ (Claude)  │   └────────────────────┘
                                 │ none      └───────────┘          │ ungrounded
                                 ▼ relevant                         ▼
                           "I don't know" (no hallucination)   abstain / flag
```

## Features

- **Pluggable embeddings**: Bedrock Titan Text Embeddings v2 in production, and a deterministic hashing embedder for offline runs and CI
- **FAISS** inner-product index on normalized vectors (cosine similarity), with save and load
- **LangGraph workflow** that filters low-relevance chunks before generation and **abstains** instead of guessing
- **Hallucination controls**: answers must cite sources, and a groundedness check flags any answer not supported by the retrieved text
- **Evaluation harness** (`eval/run_eval.py`) that reports answer accuracy, retrieval hit rate, p50/p95 latency and estimated cost per query
- **Observability**: Prometheus counters and histograms at `/metrics`
- **Reusable agent templates** in `templates/` for building new LangGraph agents

## Quick start

```bash
pip install -r requirements.txt

# Build an index from the sample docs (offline embedder)
python -m rag.ingest --source data/docs --index .index

# Ask a question
python -m rag.cli --index .index "How do I roll back a Kubernetes deployment?"

# Serve the API
uvicorn rag.api:app --reload      # POST /ask, GET /metrics

# Evaluate
python -m eval.run_eval --index .index --dataset eval/dataset.jsonl
```

### Use AWS Bedrock

```bash
export EMBEDDINGS_PROVIDER=bedrock LLM_PROVIDER=bedrock AWS_REGION=us-east-1
python -m rag.ingest --source s3://my-bucket/handbook/ --index .index
```

## Baseline results

The offline configuration (hashing embedder and extractive generator) on `eval/dataset.jsonl` scores **accuracy 0.83** and **retrieval hit rate 1.0**, and it correctly abstains on the out-of-domain question. CI fails the build if accuracy drops below 0.8. The one miss (an Argo CD–specific rollback question) is the kind of case a Bedrock generator handles better than plain sentence extraction.

## Layout

| Path | Purpose |
|------|---------|
| `rag/loaders.py` | Local folder and S3 document loaders |
| `rag/chunking.py` | Recursive, overlap-aware text splitter |
| `rag/embeddings.py` | Bedrock and hashing embedders |
| `rag/store.py` | FAISS vector store |
| `rag/graph.py` | LangGraph retrieve → grade → generate → verify workflow |
| `rag/llm.py` | Bedrock Claude and extractive fallback generators |
| `rag/api.py` | FastAPI service with Prometheus metrics |
| `eval/` | Evaluation dataset and harness |
| `templates/` | Reusable LangGraph agent template |

## Tech

LangChain · LangGraph · AWS Bedrock · FAISS · Python · FastAPI · AWS S3 · Prometheus
