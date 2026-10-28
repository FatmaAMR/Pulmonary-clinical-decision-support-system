import json
import sqlite3
from typing import List, Dict, Any, Optional
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from core.interfaces import BaseDocumentStore
from core.schemas import StoredRawFile, RawDocument, DocumentChunk, SourceMetadata
from core.registry import registry


def _metadata_to_dict(meta: SourceMetadata) -> dict:
    return {
        "source_name": meta.source_name,
        "version_date": meta.version_date,
        "source_link": meta.source_link,
        "file_format": meta.file_format,
        "page_or_section": meta.page_or_section,
        "extra": meta.extra,
    }


def _dict_to_metadata(data: dict) -> SourceMetadata:
    return SourceMetadata(
        source_name=data["source_name"],
        version_date=data["version_date"],
        source_link=data["source_link"],
        file_format=data["file_format"],
        page_or_section=data.get("page_or_section"),
        extra=data.get("extra", {}),
    )


# Object Pool + Repository Pattern - ususes live TCP connections via ConnectionPool for high concurrency
@registry.register_document_store("postgres_store")
class PostgresDocumentStore(BaseDocumentStore):
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10, use_pool: bool = True):
        self.dsn = dsn
        self.use_pool = use_pool
        self._pool: Optional[ConnectionPool] = None
        if self.use_pool:
            self._pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=min_size,
                max_size=max_size,
                kwargs={"row_factory": dict_row},
                open=True,
            )
        self._create_tables()

    def _get_connection(self):
        if self._pool is not None:
            return self._pool.connection()
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

    def _create_tables(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS raw_files (
                        file_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_format TEXT NOT NULL,
                        content_hash TEXT UNIQUE NOT NULL,
                        raw_bytes BYTEA NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_files_content_hash 
                    ON raw_files(content_hash);
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS parsed_documents (
                        doc_id TEXT PRIMARY KEY,
                        file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES parsed_documents(doc_id) ON DELETE CASCADE,
                        file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
                        token_estimate INTEGER NOT NULL,
                        content_type TEXT NOT NULL,
                        config_hash TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_file_config 
                    ON document_chunks(file_id, config_hash);
                """)
            conn.commit()

    def check_health(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS files_count FROM raw_files;")
                files_cnt = cur.fetchone()["files_count"]
                cur.execute("SELECT COUNT(*) AS chunks_count FROM document_chunks;")
                chunks_cnt = cur.fetchone()["chunks_count"]
        return {
            "status": "healthy",
            "backend": "postgresql",
            "raw_files_count": files_cnt,
            "total_chunks_count": chunks_cnt,
        }

    def save_raw_file(self, raw_file: StoredRawFile) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO raw_files (file_id, filename, file_format, content_hash, raw_bytes, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO UPDATE 
                    SET content_hash = EXCLUDED.content_hash
                    RETURNING file_id;
                    """,
                    (
                        raw_file.file_id,
                        raw_file.filename,
                        raw_file.file_format,
                        raw_file.content_hash,
                        raw_file.raw_bytes,
                        Jsonb(_metadata_to_dict(raw_file.metadata)),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            return row["file_id"]

    def get_raw_file(self, file_id: str) -> Optional[StoredRawFile]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT file_id, filename, file_format, raw_bytes, metadata FROM raw_files WHERE file_id = %s;",
                    (file_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return StoredRawFile(
                    file_id=row["file_id"],
                    filename=row["filename"],
                    file_format=row["file_format"],
                    raw_bytes=bytes(row["raw_bytes"]),
                    metadata=_dict_to_metadata(row["metadata"]),
                )

    def list_raw_files(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT r.file_id, r.filename, r.file_format, r.content_hash,
                           OCTET_LENGTH(r.raw_bytes) AS size_bytes, r.metadata,
                           COUNT(c.chunk_id) AS chunk_count
                    FROM raw_files r
                    LEFT JOIN document_chunks c ON r.file_id = c.file_id
                    GROUP BY r.file_id
                    ORDER BY r.created_at DESC;
                """)
                return [dict(r) for r in cur.fetchall()]

    def delete_file(self, file_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM raw_files WHERE file_id = %s RETURNING file_id;", (file_id,))
                deleted = cur.fetchone() is not None
            conn.commit()
            return deleted

    def save_parsed_documents(self, documents: List[RawDocument]) -> None:
        if not documents:
            return
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    cur.execute(
                        """
                        INSERT INTO parsed_documents (doc_id, file_id, content, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET content = EXCLUDED.content, metadata = EXCLUDED.metadata;
                        """,
                        (doc.doc_id, doc.file_id, doc.content, Jsonb(_metadata_to_dict(doc.metadata))),
                    )
            conn.commit()

    def get_parsed_documents_by_file(self, file_id: str) -> List[RawDocument]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id, file_id, content, metadata FROM parsed_documents WHERE file_id = %s ORDER BY created_at ASC;",
                    (file_id,),
                )
                return [
                    RawDocument(
                        doc_id=r["doc_id"],
                        file_id=r["file_id"],
                        content=r["content"],
                        metadata=_dict_to_metadata(r["metadata"]),
                    )
                    for r in cur.fetchall()
                ]

    def save_chunks(self, file_id: str, chunks: List[DocumentChunk], config_hash: str, replace_existing: bool = False) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if replace_existing:
                    cur.execute("DELETE FROM document_chunks WHERE file_id = %s;", (file_id,))
                for c in chunks:
                    cur.execute(
                        """
                        INSERT INTO document_chunks 
                        (chunk_id, doc_id, file_id, chunk_index, text, char_count, token_estimate, content_type, config_hash, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET text = EXCLUDED.text, config_hash = EXCLUDED.config_hash, metadata = EXCLUDED.metadata;
                        """,
                        (
                            c.chunk_id,
                            c.doc_id,
                            file_id,
                            c.chunk_index,
                            c.text,
                            c.char_count,
                            c.token_estimate,
                            c.content_type,
                            config_hash,
                            Jsonb(_metadata_to_dict(c.metadata)),
                        ),
                    )
            conn.commit()

    def get_chunks_by_file(self, file_id: str, config_hash: Optional[str] = None) -> List[DocumentChunk]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if config_hash:
                    cur.execute(
                        """
                        SELECT chunk_id, doc_id, file_id, chunk_index, text, content_type, config_hash, metadata
                        FROM document_chunks
                        WHERE file_id = %s AND config_hash = %s
                        ORDER BY chunk_index ASC;
                        """,
                        (file_id, config_hash),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chunk_id, doc_id, file_id, chunk_index, text, content_type, config_hash, metadata
                        FROM document_chunks
                        WHERE file_id = %s
                        ORDER BY chunk_index ASC;
                        """,
                        (file_id,),
                    )
                return [
                    DocumentChunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        file_id=r["file_id"],
                        chunk_index=r["chunk_index"],
                        text=r["text"],
                        content_type=r["content_type"],
                        config_hash=r["config_hash"],
                        metadata=_dict_to_metadata(r["metadata"]),
                    )
                    for r in cur.fetchall()
                ]


@registry.register_document_store("sqlite_store")
class SQLiteDocumentStore(BaseDocumentStore):
    def __init__(self, dsn: str = ":memory:"):
        self.dsn = dsn
        self.conn = sqlite3.connect(self.dsn, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_files (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_format TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                raw_bytes BLOB NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parsed_documents (
                doc_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES parsed_documents(doc_id) ON DELETE CASCADE,
                file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                token_estimate INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def check_health(self) -> Dict[str, Any]:
        cur = self.conn.cursor()
        files_cnt = cur.execute("SELECT COUNT(*) FROM raw_files").fetchone()[0]
        chunks_cnt = cur.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
        return {
            "status": "healthy",
            "backend": "sqlite",
            "raw_files_count": files_cnt,
            "total_chunks_count": chunks_cnt,
        }

    def save_raw_file(self, raw_file: StoredRawFile) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT file_id FROM raw_files WHERE content_hash = ?", (raw_file.content_hash,))
        existing = cur.fetchone()
        if existing:
            return existing[0]

        cur.execute(
            "INSERT INTO raw_files VALUES (?, ?, ?, ?, ?, ?)",
            (
                raw_file.file_id,
                raw_file.filename,
                raw_file.file_format,
                raw_file.content_hash,
                raw_file.raw_bytes,
                json.dumps(_metadata_to_dict(raw_file.metadata)),
            ),
        )
        self.conn.commit()
        return raw_file.file_id

    def get_raw_file(self, file_id: str) -> Optional[StoredRawFile]:
        cur = self.conn.cursor()
        cur.execute("SELECT file_id, filename, file_format, raw_bytes, metadata FROM raw_files WHERE file_id = ?", (file_id,))
        row = cur.fetchone()
        if not row:
            return None
        return StoredRawFile(
            file_id=row[0],
            filename=row[1],
            file_format=row[2],
            raw_bytes=row[3],
            metadata=_dict_to_metadata(json.loads(row[4])),
        )

    def list_raw_files(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT r.file_id, r.filename, r.file_format, r.content_hash,
                   LENGTH(r.raw_bytes) AS size_bytes, r.metadata,
                   COUNT(c.chunk_id) AS chunk_count
            FROM raw_files r
            LEFT JOIN document_chunks c ON r.file_id = c.file_id
            GROUP BY r.file_id
        """)
        rows = cur.fetchall()
        return [
            {
                "file_id": r[0],
                "filename": r[1],
                "file_format": r[2],
                "content_hash": r[3],
                "size_bytes": r[4],
                "metadata": json.loads(r[5]),
                "chunk_count": r[6],
            }
            for r in rows
        ]

    def delete_file(self, file_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM raw_files WHERE file_id = ?", (file_id,))
        deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def save_parsed_documents(self, documents: List[RawDocument]) -> None:
        cur = self.conn.cursor()
        for doc in documents:
            cur.execute(
                "INSERT OR REPLACE INTO parsed_documents VALUES (?, ?, ?, ?)",
                (doc.doc_id, doc.file_id or "unlinked", doc.content, json.dumps(_metadata_to_dict(doc.metadata))),
            )
        self.conn.commit()

    def get_parsed_documents_by_file(self, file_id: str) -> List[RawDocument]:
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id, file_id, content, metadata FROM parsed_documents WHERE file_id = ?", (file_id,))
        return [
            RawDocument(doc_id=r[0], file_id=r[1], content=r[2], metadata=_dict_to_metadata(json.loads(r[3])))
            for r in cur.fetchall()
        ]

    def save_chunks(self, file_id: str, chunks: List[DocumentChunk], config_hash: str, replace_existing: bool = False) -> None:
        cur = self.conn.cursor()
        if replace_existing:
            cur.execute("DELETE FROM document_chunks WHERE file_id = ?", (file_id,))
        for c in chunks:
            cur.execute(
                "INSERT OR REPLACE INTO document_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c.chunk_id,
                    c.doc_id,
                    file_id,
                    c.chunk_index,
                    c.text,
                    c.char_count,
                    c.token_estimate,
                    c.content_type,
                    config_hash,
                    json.dumps(_metadata_to_dict(c.metadata)),
                ),
            )
        self.conn.commit()

    def get_chunks_by_file(self, file_id: str, config_hash: Optional[str] = None) -> List[DocumentChunk]:
        cur = self.conn.cursor()
        if config_hash:
            cur.execute(
                """
                SELECT chunk_id, doc_id, file_id, chunk_index, text, content_type, config_hash, metadata
                FROM document_chunks WHERE file_id = ? AND config_hash = ? ORDER BY chunk_index ASC
                """,
                (file_id, config_hash),
            )
        else:
            cur.execute(
                """
                SELECT chunk_id, doc_id, file_id, chunk_index, text, content_type, config_hash, metadata
                FROM document_chunks WHERE file_id = ? ORDER BY chunk_index ASC
                """,
                (file_id,),
            )
        return [
            DocumentChunk(
                chunk_id=r[0],
                doc_id=r[1],
                file_id=r[2],
                chunk_index=r[3],
                text=r[4],
                content_type=r[5],
                config_hash=r[6],
                metadata=_dict_to_metadata(json.loads(r[7])),
            )
            for r in cur.fetchall()
        ]
    