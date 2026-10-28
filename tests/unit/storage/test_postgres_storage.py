import os
import json
import pytest
from unittest.mock import MagicMock, patch
from core.schemas import SourceMetadata, StoredRawFile
from core.registry import registry
import components.storage
import components.parsers


def test_both_postgres_and_sqlite_stores_registered():
    pg_cls = registry.get_document_store("postgres_store")
    sq_cls = registry.get_document_store("sqlite_store")
    assert pg_cls.__name__ == "PostgresDocumentStore"
    assert sq_cls.__name__ == "SQLiteDocumentStore"


@patch("components.storage.postgres_document_store.psycopg.connect")
def test_postgres_store_executes_bytea_and_jsonb_upsert(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store_cls = registry.get_document_store("postgres_store")
    store = store_cls(dsn="postgresql://user:pass@localhost:5432/rag_db", use_pool=False)

    meta = SourceMetadata(
        source_name="WHO Chest X-Ray Guideline",
        version_date="2025",
        source_link="https://who.int/cxr",
        file_format="pdf",
    )
    raw_file = StoredRawFile(
        filename="who_cxr.pdf",
        file_format=".pdf",
        raw_bytes=b"%PDF-1.4 clinical content",
        metadata=meta,
    )

    mock_cursor.fetchone.return_value = {"file_id": raw_file.file_id}
    returned_id = store.save_raw_file(raw_file)

    assert returned_id == raw_file.file_id
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert "ON CONFLICT (content_hash)" in executed_sql
    assert "RETURNING file_id" in executed_sql


def test_end_to_end_raw_blob_to_parsed_docs_contract():
        dsn = os.getenv("TEST_POSTGRES_DSN")
        if dsn:
            store = registry.get_document_store("postgres_store")(dsn=dsn)
        else:
            store = registry.get_document_store("sqlite_store")(dsn=":memory:")

        payload = [
            {
                "text": "Atelectasis and pleural effusion correlation with clinical history.",
                "section": "Section 3.2",
                "source_name": "ACR Thoracic Guideline",
                "version_date": "2025-02",
            }
        ]
        raw_bytes = json.dumps(payload).encode("utf-8")
        meta = SourceMetadata(
            source_name="ACR Thoracic Guideline",
            version_date="2025-02",
            source_link="https://acr.org",
            file_format="json",
        )
        
        raw_file = StoredRawFile(filename="acr.json", file_format=".json", raw_bytes=raw_bytes, metadata=meta)
        
        # SWE Fix: Clean up environment before testing to avoid accumulating records across test runs
        store.delete_file(raw_file.file_id)

        file_id = store.save_raw_file(raw_file)

        duplicate_file = StoredRawFile(filename="acr_dup.json", file_format=".json", raw_bytes=raw_bytes, metadata=meta)
        assert store.save_raw_file(duplicate_file) == file_id

        fetched_blob = store.get_raw_file(file_id)
        parser = registry.get_parser(fetched_blob.file_format)()
        parsed_docs = parser.parse(
            raw_bytes=fetched_blob.raw_bytes,
            base_metadata={"source_name": fetched_blob.metadata.source_name},
            file_id=file_id,
        )

        # Clear existing parsed documents for this file to ensure exactly 1 record is asserted
        if hasattr(store, "_get_connection") and dsn:
            with store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM parsed_documents WHERE file_id = %s;", (file_id,))
                conn.commit()

        store.save_parsed_documents(parsed_docs)
        saved_docs = store.get_parsed_docs_by_file(file_id) if hasattr(store, "get_parsed_docs_by_file") else store.get_parsed_documents_by_file(file_id)

        assert len(saved_docs) == 1
