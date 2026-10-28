from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Optional
from api.dependencies import get_pipeline
from api.schemas import ChunkingRequest, ChunkingResponse, ChunkListResponse
from pipeline.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["Chunks Inspector"])


@router.post("/{file_id}/chunk", response_model=ChunkingResponse, status_code=status.HTTP_200_OK)
def trigger_document_chunking(
    file_id: str,
    req: ChunkingRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline)
):
    try:
        res = pipeline.run_chunking_for_file(
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{file_id}/chunks", response_model=ChunkListResponse, status_code=status.HTTP_200_OK)
def inspect_document_chunks(
    file_id: str,
    limit: int = Query(10, ge=1, le=100, description="Max chunks to return"),
    offset: int = Query(0, ge=0, description="Chunks to skip"),
    content_type: Optional[str] = Query(None, description="Filter by prose, table, list, or code"),
    section_contains: Optional[str] = Query(None, description="Filter by section name"),
    pipeline: IngestionPipeline = Depends(get_pipeline)
):
    raw_file = pipeline.store.get_raw_file(file_id)
    if not raw_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{file_id}' not found.")

    # In a massive DB, this logic is pushed to SQL using LIMIT/OFFSET. 
    # For RAG inspection APIs, in-memory slicing is perfectly standard.
    chunks = pipeline.store.get_chunks_by_file(file_id)
    total_in_db = len(chunks)

    # Apply Filters
    if content_type:
        chunks = [c for c in chunks if c.content_type == content_type]
    if section_contains:
        chunks = [c for c in chunks if c.metadata.page_or_section and section_contains.lower() in c.metadata.page_or_section.lower()]

    # Apply Pagination
    paginated_chunks = chunks[offset : offset + limit]

    return {
        "file_id": file_id,
        "filename": raw_file.filename,
        "total_chunks_in_db": total_in_db,
        "returned_count": len(paginated_chunks),
        "limit": limit,
        "offset": offset,
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
            for c in paginated_chunks
        ],
    }