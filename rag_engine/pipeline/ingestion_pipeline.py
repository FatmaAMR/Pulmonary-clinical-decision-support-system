import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.schemas import SourceMetadata, StoredRawFile, RawDocument, DocumentChunk
from core.interfaces import BaseDocumentStore, BaseChunker
from core.registry import registry
from components.normalizers import NormalizerChain
import components.storage
import components.parsers
import components.chunkers


# Facade & Template Method - coordinates Storage, Parsing, Normalization, and Idempotent Chunking
class IngestionPipeline:
    def __init__(self, config: Dict[str, Any], store: Optional[BaseDocumentStore] = None):
        self.config = config
        if store is not None:
            self.store = store
        else:
            store_cfg = config["document_store"]
            store_cls = registry.get_document_store(store_cfg["provider"])
            self.store = store_cls(**store_cfg.get("params", {}))

    def _compute_config_hash(self, normalizers_cfg: List[Dict[str, Any]], chunker_cfg: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"normalizers": normalizers_cfg, "chunker": chunker_cfg},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def upload_raw_document(
        self,
        filename: str,
        raw_bytes: bytes,
        source_name: str,
        version_date: str = "N/A",
        source_link: str = "db://raw_files",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not raw_bytes or not raw_bytes.strip():
            raise ValueError("Uploaded file bytes cannot be empty.")

        ext = Path(filename).suffix.lower()
        if not ext:
            raise ValueError(f"Filename '{filename}' must have a valid extension.")

        # validate that a parser exists for this format before storing
        registry.get_parser(ext)

        meta = SourceMetadata(
            source_name=source_name,
            version_date=version_date,
            source_link=source_link,
            file_format=ext.lstrip("."),
            extra=extra_metadata or {},
        )
        raw_file = StoredRawFile(
            filename=filename,
            file_format=ext,
            raw_bytes=raw_bytes,
            metadata=meta,
        )

        existing_files = {f["content_hash"]: f["file_id"] for f in self.store.list_raw_files()}
        is_duplicate = raw_file.content_hash in existing_files

        saved_file_id = self.store.save_raw_file(raw_file)
        return {
            "file_id": saved_file_id,
            "filename": filename,
            "content_hash": raw_file.content_hash,
            "deduplicated": is_duplicate,
        }

    def run_chunking_for_file(
        self,
        file_id: str,
        force_rechunk: bool = False,
        chunker_override: Optional[Dict[str, Any]] = None,
        normalizers_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        stored_file = self.store.get_raw_file(file_id)
        if not stored_file:
            raise KeyError(f"File with file_id '{file_id}' does not exist in storage.")

        ingestion_cfg = self.config.get("ingestion", {})
        norm_cfg = normalizers_override if normalizers_override is not None else ingestion_cfg.get("normalizers", [])
        chunk_cfg = chunker_override if chunker_override is not None else ingestion_cfg.get("chunker", {"name": "recursive_chunker", "params": {}})

        config_hash = self._compute_config_hash(norm_cfg, chunk_cfg)

        # idempotency Guard - Skip heavy parsing/chunking if exact config was already executed.
        if not force_rechunk:
            existing_chunks = self.store.get_chunks_by_file(file_id, config_hash=config_hash)
            if existing_chunks:
                return {
                    "file_id": file_id,
                    "status": "cached_skip",
                    "config_hash": config_hash,
                    "chunk_count": len(existing_chunks),
                    "chunks": existing_chunks,
                }

        # Step 1: Parse Raw Bytes into RawDocuments
        parser_cls = registry.get_parser(stored_file.file_format)
        raw_docs = parser_cls().parse(
            raw_bytes=stored_file.raw_bytes,
            base_metadata={
                "source_name": stored_file.metadata.source_name,
                "version_date": stored_file.metadata.version_date,
                "source_link": stored_file.metadata.source_link,
                "extra": stored_file.metadata.extra,
            },
            file_id=file_id,
        )

        # Step 2: Apply Normalizer Chain
        normalizer_chain = NormalizerChain(norm_cfg)
        normalized_docs = normalizer_chain.apply_to_documents(raw_docs)
        self.store.save_parsed_documents(normalized_docs)

        # Step 3: Execute Chunker Strategy
        chunker_cls = registry.get_chunker(chunk_cfg["name"])
        chunker: BaseChunker = chunker_cls(**chunk_cfg.get("params", {}))

        all_chunks: List[DocumentChunk] = []
        global_idx = 0
        for doc in normalized_docs:
            doc_chunks = chunker.chunk(doc)
            for c in doc_chunks:
                c.file_id = file_id
                c.chunk_index = global_idx
                c.config_hash = config_hash
                all_chunks.append(c)
                global_idx += 1

        # Step 4: Persist Chunks Atomically
        self.store.save_chunks(
            file_id=file_id,
            chunks=all_chunks,
            config_hash=config_hash,
            replace_existing=True,
        )

        return {
            "file_id": file_id,
            "status": "newly_chunked",
            "config_hash": config_hash,
            "chunk_count": len(all_chunks),
            "chunks": all_chunks,
        }