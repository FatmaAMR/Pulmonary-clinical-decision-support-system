from fastapi import Request
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.pipeline


def get_retrieval_pipeline(request: Request) -> RetrievalPipeline:
    return request.app.state.retrieval_pipeline


def get_rag_pipeline(request: Request) -> EndToEndRAGPipeline:
    return request.app.state.rag_pipeline
