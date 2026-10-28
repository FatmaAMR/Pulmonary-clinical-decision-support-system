import pytest
from fastapi.testclient import TestClient
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from api.main import create_app, load_default_config


@pytest.fixture
def client():
    cfg = load_default_config()
    sqlite_store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    in_mem_vs = registry.get_vector_store("in_memory_vector_store")()
    embedder = registry.get_embedder("hash_embedder")(dim=384)
    reranker = registry.get_reranker("hybrid_overlap_reranker")()

    ingest_pipe = IngestionPipeline(config=cfg, store=sqlite_store)
    ret_pipe = RetrievalPipeline(
        config=cfg,
        document_store=sqlite_store,
        embedder=embedder,
        vector_store=in_mem_vs,
        reranker=reranker,
    )
    app = create_app(pipeline=ingest_pipe, retrieval_pipeline=ret_pipe)
    return TestClient(app)


def test_end_to_end_hybrid_retrieval_and_evaluation_metrics(client):
    corpus_md = (
        "# 4.1 Semantic vs. Lexical Retrieval\n"
        "Systems run a parallel lexical search using BM25 based on TF-IDF. "
        "Results from semantic and lexical search are fused using Reciprocal Rank Fusion (RRF).\n\n"
        "# 5. Vector Databases and Distance Metrics\n"
        "The most prevalent ANN algorithm is HNSW (Hierarchical Navigable Small World), "
        "which constructs a multi-layered graph for fast routing.\n\n"
        "# 8. Stress Testing with Edge Cases\n"
        "Consider alphanumeric codes: TX-9902-A, REV. B, and serial number 84729-XXL-01. "
        "Inline math variation is ±15% when temperature T >= 100°C."
    ).encode("utf-8")

    up_res = client.post(
        "/documents",
        files={"file": ("rag_corpus.md", corpus_md, "text/markdown")},
        data={"source_name": "RAG Evaluation Corpus", "auto_chunk": "true"},
    )
    assert up_res.status_code == 201
    file_id = up_res.json()["file_id"]

    # 1. Index chunks into Vector Store
    idx_res = client.post(f"/documents/{file_id}/index")
    assert idx_res.status_code == 200
    assert idx_res.json()["indexed_chunks_count"] == 3

    # 2. Test Hybrid Search for exact serial number & alphanumeric code
    search_res = client.post(
        "/retrieval/search",
        json={"query": "What is serial number 84729-XXL-01 and code TX-9902-A?", "top_k": 3, "mode": "hybrid"},
    ).json()
    assert search_res["abstained"] is False
    assert "84729-XXL-01" in search_res["results"][0]["text"]

    # 3. Test Selective Abstention Gate on out-of-domain query
    abstain_res = client.post(
        "/retrieval/search",
        json={"query": "What is the surgical dosage of pediatric amoxicillin?", "top_k": 3},
    ).json()
    assert abstain_res["abstained"] is True

    # 4. Run Automated Retrieval Benchmark Suite (Hit Rate, MRR, NDCG, Abstention Accuracy)
    eval_payload = {
        "top_k": 3,
        "mode": "hybrid",
        "file_id_filter": file_id,
        "test_cases": [
            {
                "query": "Which algorithm fuses BM25 lexical search and semantic search?",
                "expected_keywords": ["Reciprocal Rank Fusion", "RRF"],
                "should_abstain": False,
            },
            {
                "query": "How does HNSW graph routing work in vector databases?",
                "expected_keywords": ["Hierarchical Navigable Small World", "multi-layered graph"],
                "should_abstain": False,
            },
            {
                "query": "Find alphanumeric code TX-9902-A and serial number 84729-XXL-01",
                "expected_keywords": ["TX-9902-A", "84729-XXL-01"],
                "should_abstain": False,
            },
            {
                "query": "How to bake chocolate croissants in a convection oven?",
                "expected_keywords": [],
                "should_abstain": True,
            },
        ],
    }
    eval_res = client.post("/retrieval/evaluate", json=eval_payload)
    assert eval_res.status_code == 200
    metrics = eval_res.json()["metrics"]

    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr_score"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
