import pytest
from fastapi.testclient import TestClient
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from api.app import create_app, load_default_config


@pytest.fixture
def client():
    cfg = load_default_config()
    sqlite_store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    pipeline = IngestionPipeline(config=cfg, store=sqlite_store)
    app = create_app(pipeline=pipeline)
    return TestClient(app)


def test_health_check_and_full_document_crud_lifecycle(client):
    # 1. Check initial DB health
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["raw_files_count"] == 0
    assert health["total_chunks_count"] == 0

    # 2. Upload a multi-section document with prose, list, and table
    sample_md = (
        "# 1. Diagnostic Protocol\n"
        "Perform initial CXR evaluation and correlate with SpO2 levels.\n\n"
        "## 2. Differential Checklist\n"
        "- Check for pleural effusion\n"
        "- Rule out lobar pneumonia\n\n"
        "## 3. Severity Grading Table\n"
        "| Grade | Finding |\n"
        "| --- | --- |\n"
        "| High | Tension Pneumothorax |"
    ).encode("utf-8")

    res = client.post(
        "/documents",
        files={"file": ("protocol.md", sample_md, "text/markdown")},
        data={
            "source_name": "Thoracic Guideline",
            "version_date": "2026-01",
            "source_link": "[https://guidelines.org/thoracic](https://guidelines.org/thoracic)",
            "auto_chunk": "true",
        },
    )
    assert res.status_code == 201
    data = res.json()
    file_id = data["file_id"]
    assert data["deduplicated"] is False
    assert data["chunking"]["chunking_status"] == "newly_chunked"
    assert data["chunking"]["chunk_count"] == 3

    # 3. Inspect stored chunks and verify content_type detection & abbreviation expansion
    chunks_res = client.get(f"/documents/{file_id}/chunks").json()
    assert chunks_res["total_chunks_returned"] == 3
    chunks = chunks_res["chunks"]

    assert chunks[0]["content_type"] == "prose"
    assert "CXR (Chest X-ray)" in chunks[0]["text"]
    assert "1. Diagnostic Protocol" in chunks[0]["metadata"]["page_or_section"]

    assert chunks[1]["content_type"] == "list"
    assert chunks[2]["content_type"] == "table"

    # Filter chunks by content_type=table
    table_chunks = client.get(f"/documents/{file_id}/chunks?content_type=table").json()
    assert table_chunks["total_chunks_returned"] == 1

    # 4. Test Idempotency Guard: Re-triggering chunking with same config must skip re-computation
    rechunk_same = client.post(
        f"/documents/{file_id}/chunk",
        json={"force_rechunk": False},
    ).json()
    assert rechunk_same["status"] == "cached_skip"
    assert rechunk_same["chunk_count"] == 3

    # 5. Test Dynamic Override: Changing chunk_size forces re-chunking and updates DB chunks
    rechunk_override = client.post(
        f"/documents/{file_id}/chunk",
        json={
            "force_rechunk": False,
            "chunker_override": {
                "name": "recursive_chunker",
                "params": {"chunk_size": 70, "chunk_overlap": 10},
            },
        },
    ).json()
    assert rechunk_override["status"] == "newly_chunked"
    assert rechunk_override["chunk_count"] > 3

    # 6. Test Cascading Delete: Deleting the document removes raw file, parsed docs, and all chunks
    del_res = client.delete(f"/documents/{file_id}")
    assert del_res.status_code == 200

    health_after = client.get("/health").json()
    assert health_after["raw_files_count"] == 0
    assert health_after["total_chunks_count"] == 0