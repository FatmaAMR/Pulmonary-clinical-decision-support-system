import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel

from pipeline.ingestion_pipeline import IngestionPipeline


class ChunkingRequest(BaseModel):
    force_rechunk: bool = False
    chunker_override: Optional[Dict[str, Any]] = None
    normalizers_override: Optional[List[Dict[str, Any]]] = None


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # allow environment variable override for DSN or test store
    env_provider = os.getenv("RAG_STORE_PROVIDER")
    env_dsn = os.getenv("RAG_DB_DSN")
    if env_provider:
        cfg["document_store"]["provider"] = env_provider
    if env_dsn:
        cfg["document_store"]["params"]["dsn"] = env_dsn
    return cfg


def create_app(pipeline: Optional[IngestionPipeline] = None) -> FastAPI:
    app = FastAPI(
        title="Modular RAG Engine - Storage & Chunking Inspector API",
        version="1.0.0",
    )

    active_pipeline = pipeline or IngestionPipeline(load_default_config())

    @app.get("/health")
    def check_health():
        try:
            return active_pipeline.store.check_health()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}")

    @app.post("/documents", status_code=201)
    async def upload_document(
        file: UploadFile = File(...),
        source_name: str = Form(...),
        version_date: str = Form("N/A"),
        source_link: str = Form("db://raw_files"),
        auto_chunk: bool = Form(True),
    ):
        raw_bytes = await file.read()
        try:
            upload_res = active_pipeline.upload_raw_document(
                filename=file.filename or "uploaded.txt",
                raw_bytes=raw_bytes,
                source_name=source_name,
                version_date=version_date,
                source_link=source_link,
            )
            chunk_info = None
            if auto_chunk:
                chunk_res = active_pipeline.run_chunking_for_file(upload_res["file_id"], force_rechunk=False)
                chunk_info = {
                    "chunking_status": chunk_res["status"],
                    "config_hash": chunk_res["config_hash"],
                    "chunk_count": chunk_res["chunk_count"],
                }
            return {**upload_res, "chunking": chunk_info}
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/documents")
    def list_documents():
        return {"documents": active_pipeline.store.list_raw_files()}

    @app.get("/documents/{file_id}")
    def get_document_details(file_id: str):
        raw_file = active_pipeline.store.get_raw_file(file_id)
        if not raw_file:
            raise HTTPException(status_code=404, detail=f"Document '{file_id}' not found.")

        parsed_docs = active_pipeline.store.get_parsed_documents_by_file(file_id)
        chunks = active_pipeline.store.get_chunks_by_file(file_id)

        return {
            "file_id": raw_file.file_id,
            "filename": raw_file.filename,
            "file_format": raw_file.file_format,
            "content_hash": raw_file.content_hash,
            "size_bytes": len(raw_file.raw_bytes),
            "metadata": raw_file.metadata,
            "parsed_documents_count": len(parsed_docs),
            "chunks_count": len(chunks),
        }

    @app.post("/documents/{file_id}/chunk")
    def trigger_document_chunking(file_id: str, req: ChunkingRequest):
        try:
            res = active_pipeline.run_chunking_for_file(
                file_id=file_id,
                force_rechunk=req.force_rechunk,
                chunker_override=req.chunker_override,
                normalizers_override=req.normalizers_override,
            )
            return {
                "file_id": res["file_id"],
                "status": res["status"],
                "config_hash": res["config_hash"],
                "chunk_count": res["chunk_count"],
                "chunks_preview": [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                        "char_count": c.char_count,
                        "token_estimate": c.token_estimate,
                        "content_type": c.content_type,
                        "section": c.metadata.page_or_section,
                        "text": c.text,
                    }
                    for c in res["chunks"]
                ],
            }
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/documents/{file_id}/chunks")
    def inspect_document_chunks(
        file_id: str,
        content_type: Optional[str] = Query(None, description="Filter by prose, table, list, or code"),
    ):
        raw_file = active_pipeline.store.get_raw_file(file_id)
        if not raw_file:
            raise HTTPException(status_code=404, detail=f"Document '{file_id}' not found.")

        chunks = active_pipeline.store.get_chunks_by_file(file_id)
        if content_type:
            chunks = [c for c in chunks if c.content_type == content_type]

        return {
            "file_id": file_id,
            "filename": raw_file.filename,
            "total_chunks_returned": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                    "token_estimate": c.token_estimate,
                    "content_type": c.content_type,
                    "config_hash": c.config_hash,
                    "metadata": c.metadata,
                    "text": c.text,
                }
                for c in chunks
            ],
        }

    @app.delete("/documents/{file_id}")
    def delete_document(file_id: str):
        deleted = active_pipeline.store.delete_file(file_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{file_id}' not found.")
        return {"status": "deleted", "file_id": file_id}

    return app


app = create_app()