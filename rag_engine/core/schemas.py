import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid


# SWE Principle: Immutability (Value Object). Prevents accidental mutation of source provenance across pipeline stages.
@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    version_date: str
    source_link: str
    file_format: str
    page_or_section: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_name or not self.source_name.strip():
            raise ValueError("source_name cannot be empty.")
        if not self.file_format or not self.file_format.strip():
            raise ValueError("file_format must be specified.")


@dataclass(frozen=True)
class StoredRawFile:
    filename: str
    file_format: str
    raw_bytes: bytes
    metadata: SourceMetadata
    file_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_hash: str = field(init=False)

    def __post_init__(self):
        if not self.raw_bytes:
            raise ValueError("raw_bytes cannot be empty.")
        computed_hash = hashlib.sha256(self.raw_bytes).hexdigest()
        object.__setattr__(self, "content_hash", computed_hash)


@dataclass
class RawDocument:
    content: str
    metadata: SourceMetadata
    file_id: Optional[str] = None
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not isinstance(self.metadata, SourceMetadata):
            raise TypeError("metadata must be an instance of SourceMetadata.")


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: SourceMetadata
    file_id: Optional[str] = None
    chunk_index: int = 0
    char_count: int = field(init=False)
    token_estimate: int = field(init=False)
    content_type: str = "prose"
    config_hash: str = "default"
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def __post_init__(self):
        self.char_count = len(self.text)
        self.token_estimate = max(1, len(self.text.split())) if self.text.strip() else 0


@dataclass
class RetrievalResult:
    query: str
    ranked_chunks: List[DocumentChunk]
    max_confidence_score: float
    abstained: bool
    abstention_reason: Optional[str] = None
    search_mode: str = "hybrid"
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalQueryCase:
    query: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_section_substr: Optional[str] = None
    should_abstain: bool = False


@dataclass
class RetrievalEvalReport:
    total_queries: int
    hit_rate_at_k: float
    mrr_score: float
    ndcg_at_k: float
    context_precision_at_k: float
    abstention_accuracy: float
    per_query_details: List[Dict[str, Any]]

@dataclass
class GenerationResult:
    query: str
    answer: str
    sources_cited: List[Dict[str, Any]]
    abstained: bool
    model_name: str
    token_usage: Dict[str, int] = field(default_factory=dict)

@dataclass
class ScoredChunk(DocumentChunk):
    similarity_score: float = 0.0
    lexical_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
