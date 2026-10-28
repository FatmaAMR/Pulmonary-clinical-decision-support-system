# ✦ Modular RAG Engine

> A domain-agnostic, schema-driven End-to-End Retrieval-Augmented Generation (RAG) architecture engineered on strict SOLID principles, ACID persistence, and deterministic execution.

![Python Version](https://img.shields.io/badge/python-3.11%2B-1f2937.svg?style=flat-square&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/architecture-SOLID%20%7C%20Plug--and--Play-0f172a.svg?style=flat-square)
![Data Integrity](https://img.shields.io/badge/persistence-ACID%20%7C%20Idempotent-111827.svg?style=flat-square)

The **Modular RAG Engine** is a full-lifecycle document intelligence, retrieval, and grounded generation framework. Rather than coupling system logic to specific databases, file systems, or model providers, the engine treats every stage of the RAG lifecycle—parsing, normalization, chunking, relational persistence, vector indexing, reranking, context augmentation, and LLM synthesis—as an interchangeable, contract-bound plugin orchestrated via declarative configurations.

---

## Engineering Philosophy

- **Declarative Configuration & Registry Binding:** Runtime behavior is governed by `pipeline_config.yaml` and `prompts.yaml`. A Singleton Registry resolves provider keys into concrete class implementations without hard-coded dependencies.
- **Zero-Filesystem Byte-Stream Ingestion:** Raw files are processed directly from memory streams into ACID relational storage (`BYTEA`/`BLOB`), enabling stateless container execution.
- **Two-Tier Cryptographic Idempotency:**
  - *Payload Deduplication:* `SHA-256` binary hashing prevents duplicate storage of identical files and dynamically maps HTTP responses (`201 Created` vs. `200 OK`).
  - *Execution Caching:* Configuration fingerprinting (`config_hash`) caches chunking and indexing states per parameter profile, bypassing redundant compute unless explicitly overridden.
- **Unbroken Evidence Provenance:** Immutable value objects (`SourceMetadata`) propagate document origin, version timestamps, and section headers from raw ingestion through to the final LLM citation payload.
- **Selective Abstention Gate:** Retrieval scores are evaluated against strict confidence thresholds prior to LLM invocation, preventing hallucinated synthesis when supporting evidence is insufficient.

---

## End-to-End System Architecture

The diagram below depicts the complete system topology across three planes: **Control & Configuration**, **Write Path (Ingestion & Indexing)**, and **Read Path (Retrieval, Augmentation & Generation)**. 
*(Legend: `●` Implemented | `◌` Active Integration)*


<img alt="FastAPI Qdrant Retrieval Evaluation Pipeline-2026-10-25-060312" src="https://github.com/user-attachments/assets/4d35b856-9ca4-4117-b0f4-6a2678a6ee32" />

