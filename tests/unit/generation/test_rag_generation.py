import pytest
from fastapi.testclient import TestClient
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline
from api.main import create_app, load_default_config


@pytest.fixture
def rag_client():
    cfg = load_default_config()
    sqlite_store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    in_mem_vs = registry.get_vector_store("in_memory_vector_store")()
    embedder = registry.get_embedder("hash_embedder")(dim=384)
    reranker = registry.get_reranker("hybrid_overlap_reranker")()
    mock_gen = registry.get_generator("extractive_mock_generator")()

    ingest_pipe = IngestionPipeline(config=cfg, store=sqlite_store)
    ret_pipe = RetrievalPipeline(
        config=cfg,
        document_store=sqlite_store,
        embedder=embedder,
        vector_store=in_mem_vs,
        reranker=reranker,
    )
    rag_pipe = EndToEndRAGPipeline(
        config=cfg,
        retrieval_pipeline=ret_pipe,
        generator=mock_gen,
    )
    app = create_app(pipeline=ingest_pipe, retrieval_pipeline=ret_pipe, rag_pipeline=rag_pipe)
    return TestClient(app)


def test_rag_generate_endpoint_returns_cited_answer_and_abstains_on_noise(rag_client):
    doc_md = (
        "# 4.1 Semantic vs. Lexical Retrieval\n"
        "Results from semantic search and BM25 lexical search are fused together using Reciprocal Rank Fusion (RRF).\n"
    ).encode("utf-8")

    up_res = rag_client.post(
        "/documents",
        files={"file": ("rag.md", doc_md, "text/markdown")},
        data={"source_name": "RAG Corpus", "auto_chunk": "true"},
    )
    file_id = up_res.json()["file_id"]
    rag_client.post(f"/documents/{file_id}/index")

    # 1. Valid in-domain question -> should answer and cite [Source 1]
    ans_res = rag_client.post(
        "/rag/generate",
        json={"query": "How are semantic search and BM25 lexical search fused together?", "top_k": 2},
    )
    assert ans_res.status_code == 200
    body = ans_res.json()
    assert body["abstained"] is False
    assert "[Source 1]" in body["answer"]
    assert body["sources_cited"][0]["cited_in_answer"] is True

    # 2. Out-of-domain question -> should trigger abstention gate without calling LLM
    abs_res = rag_client.post(
        "/rag/generate",
        json={"query": "What is the recommended surgical dosage of pediatric amoxicillin?", "top_k": 2},
    )
    assert abs_res.status_code == 200
    assert abs_res.json()["abstained"] is True
    assert abs_res.json()["sources_cited"] == []
