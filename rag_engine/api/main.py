import os
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline
from core.config_loader import load_pipeline_config

from api.routers import (
    health_router,
    documents_router,
    chunks_router,
    retrieval_router,
    generation_router,
)


class RuntimeConfigUpdate(BaseModel):
    abstention_threshold: Optional[float] = None
    semantic_weight: Optional[float] = None
    lexical_weight: Optional[float] = None
    section_boost: Optional[float] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    min_citation_token_overlap: Optional[int] = None
    refusal_phrases: Optional[List[str]] = None


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    cfg = load_pipeline_config(config_path)

    provider = cfg.get("document_store", {}).get("provider")
    if provider == "sqlite_store":
        cfg.setdefault("retrieval", {})["vector_store"] = {
            "provider": "in_memory_vector_store",
            "params": {},
        }

    env_dsn = os.getenv("RAG_DB_DSN")
    if env_dsn:
        cfg.setdefault("document_store", {}).setdefault("params", {})["dsn"] = env_dsn
        if cfg.get("retrieval", {}).get("vector_store", {}).get("provider") == "pgvector_store":
            cfg["retrieval"]["vector_store"].setdefault("params", {})["dsn"] = env_dsn

    return cfg


def _mask_sensitive_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    masked = copy.deepcopy(cfg)
    try:
        if "dsn" in masked.get("document_store", {}).get("params", {}):
            masked["document_store"]["params"]["dsn"] = "***MASKED_DSN***"
        if "dsn" in masked.get("retrieval", {}).get("vector_store", {}).get("params", {}):
            masked["retrieval"]["vector_store"]["params"]["dsn"] = "***MASKED_DSN***"
    except Exception:
        pass
    return masked


def create_app(
    pipeline: Optional[IngestionPipeline] = None,
    retrieval_pipeline: Optional[RetrievalPipeline] = None,
    rag_pipeline: Optional[EndToEndRAGPipeline] = None,
) -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Engine - End-to-End API",
        version="3.0.0",
        description="Modular REST API for ingestion, hybrid retrieval, quantitative evaluation, and LLM generation.",
    )

    cfg = load_default_config()
    active_ingest = pipeline or IngestionPipeline(cfg)
    active_retrieval = retrieval_pipeline or RetrievalPipeline(
        config=cfg,
        document_store=active_ingest.store,
    )
    active_rag = rag_pipeline or EndToEndRAGPipeline(
        config=cfg,
        retrieval_pipeline=active_retrieval,
    )

    app.state.config = cfg
    app.state.pipeline = active_ingest
    app.state.retrieval_pipeline = active_retrieval
    app.state.rag_pipeline = active_rag

    app.include_router(health_router.router)
    app.include_router(documents_router.router)
    app.include_router(chunks_router.router)
    app.include_router(retrieval_router.router)
    app.include_router(generation_router.router)

    @app.get("/config", tags=["Runtime Configuration"])
    def get_active_config(request: Request):
        return _mask_sensitive_config(request.app.state.config)

    @app.patch("/config/runtime", tags=["Runtime Configuration"])
    def update_runtime_config(updates: RuntimeConfigUpdate, request: Request):
        cfg_state = request.app.state.config
        ret_pipe = request.app.state.retrieval_pipeline
        rag_pipe = request.app.state.rag_pipeline

        if updates.abstention_threshold is not None:
            ret_pipe.abstention_threshold = float(updates.abstention_threshold)
            cfg_state.setdefault("retrieval", {})["abstention_threshold"] = float(updates.abstention_threshold)

        if updates.semantic_weight is not None and hasattr(ret_pipe.reranker, "semantic_weight"):
            ret_pipe.reranker.semantic_weight = float(updates.semantic_weight)
            cfg_state.setdefault("retrieval", {}).setdefault("reranker", {}).setdefault("params", {})["semantic_weight"] = float(updates.semantic_weight)

        if updates.lexical_weight is not None and hasattr(ret_pipe.reranker, "lexical_weight"):
            ret_pipe.reranker.lexical_weight = float(updates.lexical_weight)
            cfg_state.setdefault("retrieval", {}).setdefault("reranker", {}).setdefault("params", {})["lexical_weight"] = float(updates.lexical_weight)

        if updates.section_boost is not None and hasattr(ret_pipe.reranker, "section_boost"):
            ret_pipe.reranker.section_boost = float(updates.section_boost)
            cfg_state.setdefault("retrieval", {}).setdefault("reranker", {}).setdefault("params", {})["section_boost"] = float(updates.section_boost)

        gen_params = cfg_state.setdefault("generation", {}).setdefault("params", {})
        if updates.model_name is not None:
            rag_pipe.generator.model_name = updates.model_name
            gen_params["model_name"] = updates.model_name

        if updates.temperature is not None:
            rag_pipe.generator.temperature = float(updates.temperature)
            gen_params["temperature"] = float(updates.temperature)

        if updates.max_tokens is not None:
            rag_pipe.generator.max_tokens = int(updates.max_tokens)
            gen_params["max_tokens"] = int(updates.max_tokens)

        if updates.system_prompt is not None:
            rag_pipe.generator.system_prompt = updates.system_prompt.strip()
            gen_params["system_prompt"] = updates.system_prompt.strip()

        if updates.user_prompt_template is not None:
            rag_pipe.generator.user_prompt_template = updates.user_prompt_template.strip()
            gen_params["user_prompt_template"] = updates.user_prompt_template.strip()

        guard_cfg = cfg_state.setdefault("guardrails", {})
        if updates.min_citation_token_overlap is not None:
            rag_pipe.min_citation_overlap = int(updates.min_citation_token_overlap)
            guard_cfg["min_citation_token_overlap"] = int(updates.min_citation_token_overlap)

        if updates.refusal_phrases is not None:
            rag_pipe.refusal_phrases = tuple(updates.refusal_phrases)
            guard_cfg["refusal_phrases"] = updates.refusal_phrases

        return {
            "status": "updated",
            "active_config": _mask_sensitive_config(cfg_state),
        }

    return app


app = create_app()