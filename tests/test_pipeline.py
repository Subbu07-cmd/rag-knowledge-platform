from rag.chunking import chunk_documents
from rag.embeddings import HashingEmbedder
from rag.graph import build_graph
from rag.ingest import build_index
from rag.llm import ExtractiveGenerator, groundedness
from rag.loaders import Document
from rag.store import VectorStore


def _graph():
    store = build_index("data/docs")
    return build_graph(store, HashingEmbedder(), ExtractiveGenerator())


def test_chunking_respects_size_and_overlap():
    doc = Document("word " * 1000, "x.md")
    chunks = chunk_documents([doc], size=200, overlap=20)
    assert len(chunks) > 5
    assert all(len(c.text) <= 230 for c in chunks)


def test_store_roundtrip(tmp_path):
    store = build_index("data/docs")
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded.chunks) == len(store.chunks)


def test_answers_with_sources():
    out = _graph().invoke({"question": "How do I roll back a Kubernetes deployment?"})
    assert "rollout undo" in out["answer"]
    assert any("kubernetes-rollbacks" in s for s in out["sources"])
    assert out["abstained"] is False


def test_abstains_when_out_of_domain():
    out = _graph().invoke({"question": "What is the capital of France?"})
    assert out["abstained"] is True
    assert out["sources"] == []


def test_groundedness_flags_unsupported_answer():
    chunks = chunk_documents([Document("Terraform state lives in S3 with DynamoDB locking.", "t.md")])
    assert groundedness("State lives in S3 with DynamoDB locking [1]", chunks) > 0.8
    assert groundedness("Bananas orbit Jupiter quickly", chunks) < 0.3
