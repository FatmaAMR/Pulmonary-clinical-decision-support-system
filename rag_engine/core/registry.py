from typing import Dict, Type
from core.interfaces import (
    BaseFileParser,
    BaseDocumentStore,
    BaseTextNormalizer,
    BaseChunker,
    BaseEmbedder,
    BaseVectorStore,
    BaseReranker,
    BaseGenerator,
)


class ComponentRegistry:
    def __init__(self):
        self._parsers: Dict[str, Type[BaseFileParser]] = {}
        self._document_stores: Dict[str, Type[BaseDocumentStore]] = {}
        self._normalizers: Dict[str, Type[BaseTextNormalizer]] = {}
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        self._embedders: Dict[str, Type[BaseEmbedder]] = {}
        self._vector_stores: Dict[str, Type[BaseVectorStore]] = {}
        self._rerankers: Dict[str, Type[BaseReranker]] = {}
        self._generators: Dict[str, Type[BaseGenerator]] = {}

    def register_parser(self, ext: str):
        ext_key = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        def decorator(cls: Type[BaseFileParser]):
            if not issubclass(cls, BaseFileParser):
                raise TypeError(f"{cls.__name__} must inherit from BaseFileParser")
            self._parsers[ext_key] = cls
            return cls
        return decorator

    def get_parser(self, ext: str) -> Type[BaseFileParser]:
        ext_key = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        if ext_key not in self._parsers:
            raise KeyError(f"No parser registered for extension '{ext_key}'")
        return self._parsers[ext_key]

    def register_document_store(self, name: str):
        def decorator(cls: Type[BaseDocumentStore]):
            if not issubclass(cls, BaseDocumentStore):
                raise TypeError(f"{cls.__name__} must inherit from BaseDocumentStore")
            self._document_stores[name] = cls
            return cls
        return decorator

    def get_document_store(self, name: str) -> Type[BaseDocumentStore]:
        if name not in self._document_stores:
            raise KeyError(f"No document store registered under '{name}'")
        return self._document_stores[name]

    def register_normalizer(self, name: str):
        def decorator(cls: Type[BaseTextNormalizer]):
            if not issubclass(cls, BaseTextNormalizer):
                raise TypeError(f"{cls.__name__} must inherit from BaseTextNormalizer")
            self._normalizers[name] = cls
            return cls
        return decorator

    def get_normalizer(self, name: str) -> Type[BaseTextNormalizer]:
        if name not in self._normalizers:
            raise KeyError(f"No normalizer registered under '{name}'")
        return self._normalizers[name]

    def register_chunker(self, name: str):
        def decorator(cls: Type[BaseChunker]):
            if not issubclass(cls, BaseChunker):
                raise TypeError(f"{cls.__name__} must inherit from BaseChunker")
            self._chunkers[name] = cls
            return cls
        return decorator

    def get_chunker(self, name: str) -> Type[BaseChunker]:
        if name not in self._chunkers:
            raise KeyError(f"No chunker registered under '{name}'")
        return self._chunkers[name]

    def register_embedder(self, name: str):
        def decorator(cls: Type[BaseEmbedder]):
            if not issubclass(cls, BaseEmbedder):
                raise TypeError(f"{cls.__name__} must inherit from BaseEmbedder")
            self._embedders[name] = cls
            return cls
        return decorator

    def get_embedder(self, name: str) -> Type[BaseEmbedder]:
        if name not in self._embedders:
            raise KeyError(f"No embedder registered under '{name}'")
        return self._embedders[name]

    def register_vector_store(self, name: str):
        def decorator(cls: Type[BaseVectorStore]):
            if not issubclass(cls, BaseVectorStore):
                raise TypeError(f"{cls.__name__} must inherit from BaseVectorStore")
            self._vector_stores[name] = cls
            return cls
        return decorator

    def get_vector_store(self, name: str) -> Type[BaseVectorStore]:
        if name not in self._vector_stores:
            raise KeyError(f"No vector store registered under '{name}'")
        return self._vector_stores[name]

    def register_reranker(self, name: str):
        def decorator(cls: Type[BaseReranker]):
            if not issubclass(cls, BaseReranker):
                raise TypeError(f"{cls.__name__} must inherit from BaseReranker")
            self._rerankers[name] = cls
            return cls
        return decorator

    def get_reranker(self, name: str) -> Type[BaseReranker]:
        if name not in self._rerankers:
            raise KeyError(f"No reranker registered under '{name}'")
        return self._rerankers[name]

    def register_generator(self, name: str):
        def decorator(cls: Type[BaseGenerator]):
            if not issubclass(cls, BaseGenerator):
                raise TypeError(f"{cls.__name__} must inherit from BaseGenerator")
            self._generators[name] = cls
            return cls
        return decorator

    def get_generator(self, name: str) -> Type[BaseGenerator]:
        if name not in self._generators:
            raise KeyError(f"No generator registered under '{name}'")
        return self._generators[name]


registry = ComponentRegistry()
