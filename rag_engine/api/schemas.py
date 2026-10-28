from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional


class ChunkingRequest(BaseModel):
    force_rechunk: bool = Field(default=False, description="Force re-chunking even if config matches")
    chunker_override: Optional[Dict[str, Any]] = Field(default=None, description="Override chunker settings")
    normalizers_override: Optional[List[Dict[str, Any]]] = Field(default=None, description="Override normalizer chain")


class ChunkPreview(BaseModel):
    chunk_id: str
    chunk_index: int
    char_count: int
    token_estimate: int
    content_type: str
    section: Optional[str]
    text: str


class ChunkingResponse(BaseModel):
    file_id: str
    status: str
    config_hash: str
    chunk_count: int
    chunks_preview: List[ChunkPreview]


class DocumentHealthResponse(BaseModel):
    status: str
    backend: str
    raw_files_count: int
    total_chunks_count: int


class ChunkListResponse(BaseModel):
    file_id: str
    filename: str
    total_chunks_in_db: int
    returned_count: int
    limit: int
    offset: int
    chunks: List[Dict[str, Any]]


def _clean_str_opt(v: Optional[str]) -> Optional[str]:
    if v is None or str(v).strip() in ("", "string", "null", "None"):
        return None
    return v


def _clean_float_opt(v: Any) -> Optional[float]:
    if v is None or v == "" or float(v) >= 1.0:
        return None
    return float(v)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    content_type_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    abstention_threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0, json_schema_extra={"example": None})

    _val_str = field_validator("file_id_filter", "content_type_filter", mode="before")(_clean_str_opt)
    _val_flt = field_validator("abstention_threshold_override", mode="before")(_clean_float_opt)


class EvalCaseInput(BaseModel):
    query: str
    expected_keywords: List[str] = Field(default_factory=list)
    expected_section_substr: Optional[str] = None
    should_abstain: bool = False


class RetrievalEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    test_cases: List[EvalCaseInput]


class RAGGenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question to answer using retrieved evidence")
    top_k: int = Field(default=3, ge=1, le=15, description="Number of top reranked chunks to pass to the LLM")
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    abstention_threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0, json_schema_extra={"example": None})

    _val_str = field_validator("file_id_filter", mode="before")(_clean_str_opt)
    _val_flt = field_validator("abstention_threshold_override", mode="before")(_clean_float_opt)
