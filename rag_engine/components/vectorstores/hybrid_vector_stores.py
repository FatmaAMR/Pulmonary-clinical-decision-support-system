import math
import re
from dataclasses import replace
from typing import List, Dict, Optional
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from core.interfaces import BaseVectorStore
from core.schemas import DocumentChunk, SourceMetadata
from core.registry import registry


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]+(?:-[a-z0-9\u0600-\u06ff]+)*", query.lower()))
    if not q_tokens:
        return 0.0
    t_lower = text.lower()
    t_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]+(?:-[a-z0-9\u0600-\u06ff]+)*", t_lower))

    exact_matches = sum(1 for qt in q_tokens if qt in t_tokens or qt in t_lower)
    return exact_matches / len(q_tokens)


def _apply_rrf_fusion(
    chunks: List[DocumentChunk],
    semantic_scores: Dict[str, float],
    lexical_scores: Dict[str, float],
    mode: str = "hybrid",
    rrf_k: int = 60,
) -> List[DocumentChunk]:
    if not chunks:
        return []

    sem_sorted = sorted(chunks, key=lambda c: semantic_scores.get(c.chunk_id, 0.0), reverse=True)
    lex_sorted = sorted(chunks, key=lambda c: lexical_scores.get(c.chunk_id, 0.0), reverse=True)

    sem_ranks = {c.chunk_id: rank + 1 for rank, c in enumerate(sem_sorted)}
    lex_ranks = {c.chunk_id: rank + 1 for rank, c in enumerate(lex_sorted)}

    scored_chunks: List[DocumentChunk] = []
    for c in chunks:
        cid = c.chunk_id
        s_score = round(semantic_scores.get(cid, 0.0), 4)
        l_score = round(lexical_scores.get(cid, 0.0), 4)

        if mode == "semantic":
            combined = s_score
        elif mode == "lexical":
            combined = l_score
        else:
            # Reciprocal Rank Fusion (RRF) with lexical boost when exact terms match
            rrf_sem = 1.0 / (rrf_k + sem_ranks[cid])
            rrf_lex = (1.0 / (rrf_k + lex_ranks[cid])) if l_score > 0 else 0.0
            combined = round((rrf_sem + rrf_lex) * 30.0 + (0.35 * s_score) + (0.35 * l_score), 4)

        scored_chunks.append(
            replace(
                c,
                similarity_score=s_score,
                lexical_score=l_score,
                rrf_score=combined,
            )
        )

    scored_chunks.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
    return scored_chunks


@registry.register_vector_store("in_memory_vector_store")
class InMemoryHybridVectorStore(BaseVectorStore):
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self._store: Dict[str, DocumentChunk] = {}

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        count = 0
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk '{chunk.chunk_id}' is missing its embedding vector.")
            self._store[chunk.chunk_id] = chunk
            count += 1
        return count

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        candidates = list(self._store.values())
        if file_id_filter:
            candidates = [c for c in candidates if c.file_id == file_id_filter]
        if content_type_filter:
            candidates = [c for c in candidates if c.content_type == content_type_filter]

        if not candidates:
            return []

        sem_scores = {c.chunk_id: _cosine_similarity(query_embedding, c.embedding or []) for c in candidates}
        lex_scores = {c.chunk_id: _lexical_overlap_score(query_text, c.text) for c in candidates}

        fused = _apply_rrf_fusion(candidates, sem_scores, lex_scores, mode=mode, rrf_k=self.rrf_k)
        return fused[:top_k]

    def delete_by_file_id(self, file_id: str) -> int:
        to_delete = [cid for cid, c in self._store.items() if c.file_id == file_id]
        for cid in to_delete:
            del self._store[cid]
        return len(to_delete)

    def count_vectors(self, file_id: Optional[str] = None) -> int:
        if file_id:
            return sum(1 for c in self._store.values() if c.file_id == file_id)
        return len(self._store)


# System Design Decision: Native PostgreSQL pgvector with HNSW index for scalable ANN search without external infra overhead.
@registry.register_vector_store("pgvector_store")
class PgVectorHybridStore(BaseVectorStore):
    def __init__(self, dsn: str, dimension: int = 384, rrf_k: int = 60, use_pool: bool = True):
        self.dsn = dsn
        self.dimension = dimension
        self.rrf_k = rrf_k
        self._pool: Optional[ConnectionPool] = None
        if use_pool:
            self._pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=1,
                max_size=10,
                kwargs={"row_factory": dict_row},
                open=True,
            )
        self._init_pgvector_schema()

    def _get_connection(self):
        if self._pool is not None:
            return self._pool.connection()
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _init_pgvector_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunk_vectors (
                        chunk_id TEXT PRIMARY KEY REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
                        doc_id TEXT NOT NULL,
                        file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content_type TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.dimension}) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_vectors_hnsw
                    ON chunk_vectors USING hnsw (embedding vector_cosine_ops);
                """)
            conn.commit()

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    if not c.embedding:
                        raise ValueError(f"Chunk '{c.chunk_id}' has no embedding vector.")
                    vec_literal = "[" + ",".join(f"{val:.6f}" for val in c.embedding) + "]"
                    meta_dict = {
                        "source_name": c.metadata.source_name,
                        "version_date": c.metadata.version_date,
                        "source_link": c.metadata.source_link,
                        "file_format": c.metadata.file_format,
                        "page_or_section": c.metadata.page_or_section,
                        "extra": c.metadata.extra,
                    }
                    cur.execute(
                        """
                        INSERT INTO chunk_vectors
                        (chunk_id, doc_id, file_id, chunk_index, content_type, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata;
                        """,
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.file_id,
                            c.chunk_index,
                            c.content_type,
                            c.text,
                            Jsonb(meta_dict),
                            vec_literal,
                        ),
                    )
            conn.commit()
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        vec_literal = "[" + ",".join(f"{val:.6f}" for val in query_embedding) + "]"
        where_clauses = []
        params: List[object] = [vec_literal]

        if file_id_filter:
            where_clauses.append("file_id = %s")
            params.append(file_id_filter)
        if content_type_filter:
            where_clauses.append("content_type = %s")
            params.append(content_type_filter)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        candidate_limit = max(top_k * 4, 25)
        params.append(vec_literal)
        params.append(candidate_limit)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_id, doc_id, file_id, chunk_index, content_type, text, metadata,
                           1 - (embedding <=> %s::vector) AS cosine_sim
                    FROM chunk_vectors
                    {where_sql}
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()

        if not rows:
            return []

        candidates: List[DocumentChunk] = []
        sem_scores: Dict[str, float] = {}
        lex_scores: Dict[str, float] = {}

        for r in rows:
            m = r["metadata"]
            meta = SourceMetadata(
                source_name=m["source_name"],
                version_date=m["version_date"],
                source_link=m["source_link"],
                file_format=m["file_format"],
                page_or_section=m.get("page_or_section"),
                extra=m.get("extra", {}),
            )
            chunk = DocumentChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                file_id=r["file_id"],
                chunk_index=r["chunk_index"],
                content_type=r["content_type"],
                text=r["text"],
                metadata=meta,
            )
            candidates.append(chunk)
            sem_scores[chunk.chunk_id] = float(r["cosine_sim"])
            lex_scores[chunk.chunk_id] = _lexical_overlap_score(query_text, chunk.text)

        fused = _apply_rrf_fusion(candidates, sem_scores, lex_scores, mode=mode, rrf_k=self.rrf_k)
        return fused[:top_k]

    def delete_by_file_id(self, file_id: str) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunk_vectors WHERE file_id = %s;", (file_id,))
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def count_vectors(self, file_id: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if file_id:
                    cur.execute("SELECT COUNT(*) AS cnt FROM chunk_vectors WHERE file_id = %s;", (file_id,))
                else:
                    cur.execute("SELECT COUNT(*) AS cnt FROM chunk_vectors;")
                return int(cur.fetchone()["cnt"])
