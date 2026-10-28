from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_retrieval_pipeline
from api.schemas import SearchRequest, RetrievalEvaluationRequest
from core.schemas import EvalQueryCase
from pipeline.retrieval_pipeline import RetrievalPipeline

router = APIRouter(tags=["Retrieval & Evaluation"])


@router.post("/documents/{file_id}/index", status_code=status.HTTP_200_OK)
def index_document_chunks(
    file_id: str,
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    try:
        return retrieval_pipeline.index_file_chunks(file_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/retrieval/search", status_code=status.HTTP_200_OK)
def search_chunks(
    req: SearchRequest,
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    try:
        res = retrieval_pipeline.retrieve(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            file_id_filter=req.file_id_filter,
            content_type_filter=req.content_type_filter,
            abstention_threshold_override=req.abstention_threshold_override,
        )
        return {
            "query": res.query,
            "search_mode": res.search_mode,
            "abstained": res.abstained,
            "abstention_reason": res.abstention_reason,
            "max_confidence_score": res.max_confidence_score,
            "returned_count": len(res.ranked_chunks),
            "telemetry": res.telemetry,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "file_id": c.file_id,
                    "chunk_index": c.chunk_index,
                    "content_type": c.content_type,
                    "section": c.metadata.page_or_section,
                    "similarity_score": c.similarity_score,
                    "lexical_score": c.lexical_score,
                    "rrf_score": c.rrf_score,
                    "rerank_score": c.rerank_score,
                    "text": c.text,
                }
                for c in res.ranked_chunks
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/retrieval/evaluate", status_code=status.HTTP_200_OK)
def evaluate_retrieval_pipeline(
    req: RetrievalEvaluationRequest,
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    try:
        cases = [
            EvalQueryCase(
                query=tc.query,
                expected_keywords=tc.expected_keywords,
                expected_section_substr=tc.expected_section_substr,
                should_abstain=tc.should_abstain,
            )
            for tc in req.test_cases
        ]
        report = retrieval_pipeline.evaluate_retrieval(
            test_cases=cases,
            top_k=req.top_k,
            mode=req.mode,
            file_id_filter=req.file_id_filter,
        )
        return {
            "total_queries": report.total_queries,
            "metrics": {
                "hit_rate_at_k": report.hit_rate_at_k,
                "mrr_score": report.mrr_score,
                "ndcg_at_k": report.ndcg_at_k,
                "context_precision_at_k": report.context_precision_at_k,
                "abstention_accuracy": report.abstention_accuracy,
            },
            "per_query_details": report.per_query_details,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
