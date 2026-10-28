from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Response
from api.dependencies import get_pipeline, get_retrieval_pipeline
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline

router = APIRouter(prefix="/documents", tags=["Documents Management"])


@router.post("")
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    source_name: str = Form(...),
    version_date: str = Form("N/A"),
    source_link: str = Form("db://raw_files"),
    auto_chunk: bool = Form(True),
    auto_index: bool = Form(True),
    force_reprocess: bool = Form(False),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    raw_bytes = await file.read()
    try:
        upload_res = pipeline.upload_raw_document(
            filename=file.filename or "uploaded.txt",
            raw_bytes=raw_bytes,
            source_name=source_name,
            version_date=version_date,
            source_link=source_link,
        )

        if upload_res.get("deduplicated") and not force_reprocess:
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_201_CREATED

        file_id = upload_res["file_id"]
        chunk_info = None
        index_info = None

        if auto_chunk:
            chunk_res = pipeline.run_chunking_for_file(file_id, force_rechunk=force_reprocess)
            chunk_info = {
                "chunking_status": chunk_res["status"],
                "config_hash": chunk_res["config_hash"],
                "chunk_count": chunk_res["chunk_count"],
            }

            if auto_index and chunk_res["chunk_count"] > 0:
                index_info = retrieval_pipeline.index_file_chunks(file_id)

        return {
            **upload_res,
            "chunking": chunk_info,
            "indexing": index_info,
            "ready_for_search": bool(index_info and index_info.get("indexed_chunks", 0) > 0),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("", status_code=status.HTTP_200_OK)
def list_documents(pipeline: IngestionPipeline = Depends(get_pipeline)):
    return {"documents": pipeline.store.list_raw_files()}


@router.get("/{file_id}", status_code=status.HTTP_200_OK)
def get_document_details(file_id: str, pipeline: IngestionPipeline = Depends(get_pipeline)):
    raw_file = pipeline.store.get_raw_file(file_id)
    if not raw_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{file_id}' not found.")

    parsed_docs = pipeline.store.get_parsed_documents_by_file(file_id)
    chunks = pipeline.store.get_chunks_by_file(file_id)

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


@router.delete("/{file_id}", status_code=status.HTTP_200_OK)
def delete_document(file_id: str, pipeline: IngestionPipeline = Depends(get_pipeline)):
    deleted = pipeline.store.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{file_id}' not found.")
    return {"status": "deleted", "file_id": file_id}