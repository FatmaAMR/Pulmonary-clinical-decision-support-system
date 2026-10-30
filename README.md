# Pulmora: Pediatric Pulmonary Clinical Decision Support System

![Version](https://img.shields.io/badge/version-3.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PgVector-336791.svg)

Pulmora is an extensible, AI-powered Pediatric Clinical Decision Support System (CDSS). It bridges the gap between deep learning-based medical imaging and evidence-grounded clinical guidelines. Designed with a strict "Clinician-First" philosophy, it provides pediatric pulmonologists with a unified workspace to analyze chest radiographs, track longitudinal patient records, and evaluate proposed treatment regimens using advanced Retrieval-Augmented Generation (RAG).

---

## System Capabilities

* **AI Treatment Evaluator:** Validates proposed pediatric antibiotic regimens, verifies weight-based dosages (mg/kg/day), and highlights contraindications based on retrieved medical guidelines.
* **Chest X-Ray Classification:** Integrated deep learning classifier to analyze pediatric chest radiographs (Normal vs. Pneumonia) and track imaging progression across patient visits.
* **Clinical Guideline RAG Engine:** A hybrid-search RAG pipeline (Vector + BM25 with Reciprocal Rank Fusion) that parses clinical PDF/MD guidelines, providing highly accurate, inline-cited answers to clinical queries.
* **Longitudinal Patient Dossiers:** Auto-generates patient IDs and maintains comprehensive clinical timelines, including parsed medical histories, vital signs, and radiographic records.
* **Modern Clinical Workspace:** A clean, responsive UI built with vanilla technologies, featuring interactive clinical intake forms and dynamic patient dossiers.

---

## Architecture Overview

Pulmora operates on a decoupled, microservice-friendly architecture utilizing FastAPI, PostgreSQL (PgVector), and local/remote LLM integrations.

```mermaid
graph TD
    classDef frontend fill:#1b1f32,stroke:#2b6ef2,stroke-width:2px,color:#fff;
    classDef api fill:#2563eb,stroke:#fff,stroke-width:2px,color:#fff;
    classDef rag fill:#eef2ff,stroke:#2563eb,stroke-width:2px,color:#13182b;
    classDef db fill:#f8faff,stroke:#1b1f32,stroke-width:2px,color:#13182b;
    classDef model fill:#1b1f32,stroke:#fff,stroke-width:2px,color:#fff;

    UI[Clinical Workspace UI]:::frontend

    subgraph Backend [FastAPI Gateway]
        API[API Router]:::api
    end

    subgraph Knowledge Engine [RAG Pipeline]
        Ingestion[Ingestion & Chunking]:::rag
        Retriever[Hybrid Retriever + RRF]:::rag
        Generator[LLM Generator]:::rag
    end

    subgraph Storage [Databases]
        PatientDB[(Patient Profiles DB)]:::db
        VectorDB[(PgVector Guidelines DB)]:::db
    end

    subgraph AI Models
        Classifier[CNN Pneumonia Classifier]:::model
    end

    UI <-->|REST API| API
    API -->|Upload Image| Classifier
    API <-->|CRUD Operations| PatientDB
    API -->|Upload Guidelines| Ingestion
    Ingestion -->|Embeddings| VectorDB
    API -->|Clinical Query| Retriever
    Retriever <-->|Search| VectorDB
    Retriever -->|Context Chunks| Generator
    Generator -->|Cited Answer| API

  