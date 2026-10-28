from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.schemas import StoredRawFile, RawDocument, DocumentChunk


class BaseFileParser(ABC):
    @abstractmethod
    def parse(self, raw_bytes: bytes, base_metadata: Dict[str, Any], file_id: Optional[str] = None) -> List[RawDocument]:
        pass


class BaseDocumentStore(ABC):
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save_raw_file(self, raw_file: StoredRawFile) -> str:
        pass

    @abstractmethod
    def get_raw_file(self, file_id: str) -> Optional[StoredRawFile]:
        pass

    @abstractmethod
    def list_raw_files(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        pass

    @abstractmethod
    def save_parsed_documents(self, documents: List[RawDocument]) -> None:
        pass

    @abstractmethod
    def get_parsed_documents_by_file(self, file_id: str) -> List[RawDocument]:
        pass

    @abstractmethod
    def save_chunks(self, file_id: str, chunks: List[DocumentChunk], config_hash: str, replace_existing: bool = False) -> None:
        pass

    @abstractmethod
    def get_chunks_by_file(self, file_id: str, config_hash: Optional[str] = None) -> List[DocumentChunk]:
        pass


class BaseTextNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        pass


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        pass


# Design Pattern: Adapter Pattern. Unifies local ONNX/Transformer models and external embedding APIs.
class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass


# Design Pattern: Repository Pattern. Abstracts vector indexing, ANN search, and hybrid RRF fusion.
class BaseVectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        pass

    @abstractmethod
    def delete_by_file_id(self, file_id: str) -> int:
        pass

    @abstractmethod
    def count_vectors(self, file_id: Optional[str] = None) -> int:
        pass


# Design Pattern: Strategy Pattern. Scores and re-orders candidate chunks for precision and abstention gating.
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: Optional[int] = None) -> List[DocumentChunk]:
        pass

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        pass
