from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_pipeline
from api.schemas import DocumentHealthResponse
from pipeline.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/health", tags=["System Health"])


@router.get("", response_model=DocumentHealthResponse, status_code=status.HTTP_200_OK)
def check_health(pipeline: IngestionPipeline = Depends(get_pipeline)):
    try:
        return pipeline.store.check_health()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error: {str(e)}"
        )