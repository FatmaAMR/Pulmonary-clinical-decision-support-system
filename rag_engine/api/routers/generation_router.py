from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_rag_pipeline
from api.schemas import RAGGenerateRequest
from pipeline.rag_pipeline import EndToEndRAGPipeline

router = APIRouter(tags=["RAG Generation (Groq LLM)"])


@router.get("/rag/models", status_code=status.HTTP_200_OK)
def get_available_llm_models(
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    try:
        available = (
            rag_pipeline.generator.list_available_models()
            if hasattr(rag_pipeline.generator, "list_available_models")
            else [getattr(rag_pipeline.generator, "model_name", "unknown")]
        )
        return {
            "configured_model": getattr(rag_pipeline.generator, "model_name", "unknown"),
            "available_models": available,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/rag/generate", status_code=status.HTTP_200_OK)
def generate_rag_answer(
    req: RAGGenerateRequest,
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    try:
        res = rag_pipeline.run(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            file_id_filter=req.file_id_filter,
            abstention_threshold_override=req.abstention_threshold_override,
        )
        return {
            "query": res.query,
            "abstained": res.abstained,
            "model_name": res.model_name,
            "answer": res.answer,
            "token_usage": res.token_usage,
            "sources_cited": res.sources_cited,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
