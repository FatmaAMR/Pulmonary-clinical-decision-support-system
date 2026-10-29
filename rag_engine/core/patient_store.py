import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None


class PatientRecordStore:
    def __init__(self, dsn: str = ":memory:", provider: str = "postgres_store"):
        self.dsn = dsn
        self.is_postgres = (
            provider == "postgres_store"
            and dsn.startswith("postgresql")
            and psycopg is not None
        )
        self._sqlite_conn = None
        if not self.is_postgres:
            db_path = "patients_local.db" if dsn.startswith("postgresql") else dsn
            self._sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._sqlite_conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS clinical_patients (
            patient_id TEXT PRIMARY KEY,
            age TEXT DEFAULT '',
            clinical_note TEXT DEFAULT '',
            history_file_id TEXT DEFAULT '',
            history_filename TEXT DEFAULT '',
            parsed_history_text TEXT DEFAULT '',
            last_xray_filename TEXT DEFAULT '',
            last_xray_prediction TEXT DEFAULT '',
            updated_at TEXT NOT NULL
        );
        """
        if self.is_postgres:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
                conn.commit()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(ddl)
            self._sqlite_conn.commit()

    def upsert_patient(
        self,
        patient_id: str,
        age: Optional[str] = None,
        clinical_note: Optional[str] = None,
        history_file_id: Optional[str] = None,
        history_filename: Optional[str] = None,
        parsed_history_text: Optional[str] = None,
        last_xray_filename: Optional[str] = None,
        last_xray_prediction: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = self.get_patient(patient_id) or {}
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        record = {
            "patient_id": patient_id.strip(),
            "age": age if age is not None else existing.get("age", ""),
            "clinical_note": clinical_note if clinical_note is not None else existing.get("clinical_note", ""),
            "history_file_id": history_file_id if history_file_id is not None else existing.get("history_file_id", ""),
            "history_filename": history_filename if history_filename is not None else existing.get("history_filename", ""),
            "parsed_history_text": parsed_history_text if parsed_history_text is not None else existing.get("parsed_history_text", ""),
            "last_xray_filename": last_xray_filename if last_xray_filename is not None else existing.get("last_xray_filename", ""),
            "last_xray_prediction": last_xray_prediction if last_xray_prediction is not None else existing.get("last_xray_prediction", ""),
            "updated_at": now_iso,
        }

        if self.is_postgres:
            sql = """
            INSERT INTO clinical_patients (
                patient_id, age, clinical_note, history_file_id, history_filename,
                parsed_history_text, last_xray_filename, last_xray_prediction, updated_at
            ) VALUES (
                %(patient_id)s, %(age)s, %(clinical_note)s, %(history_file_id)s, %(history_filename)s,
                %(parsed_history_text)s, %(last_xray_filename)s, %(last_xray_prediction)s, %(updated_at)s
            )
            ON CONFLICT (patient_id) DO UPDATE SET
                age = EXCLUDED.age,
                clinical_note = EXCLUDED.clinical_note,
                history_file_id = EXCLUDED.history_file_id,
                history_filename = EXCLUDED.history_filename,
                parsed_history_text = EXCLUDED.parsed_history_text,
                last_xray_filename = EXCLUDED.last_xray_filename,
                last_xray_prediction = EXCLUDED.last_xray_prediction,
                updated_at = EXCLUDED.updated_at;
            """
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, record)
                conn.commit()
        else:
            sql = """
            INSERT OR REPLACE INTO clinical_patients (
                patient_id, age, clinical_note, history_file_id, history_filename,
                parsed_history_text, last_xray_filename, last_xray_prediction, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cur = self._sqlite_conn.cursor()
            cur.execute(
                sql,
                (
                    record["patient_id"],
                    record["age"],
                    record["clinical_note"],
                    record["history_file_id"],
                    record["history_filename"],
                    record["parsed_history_text"],
                    record["last_xray_filename"],
                    record["last_xray_prediction"],
                    record["updated_at"],
                ),
            )
            self._sqlite_conn.commit()

        return record

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if self.is_postgres:
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM clinical_patients WHERE patient_id = %s", (patient_id.strip(),))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT * FROM clinical_patients WHERE patient_id = ?", (patient_id.strip(),))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_patients(self) -> List[Dict[str, Any]]:
        if self.is_postgres:
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM clinical_patients ORDER BY updated_at DESC")
                    return [dict(r) for r in cur.fetchall()]
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT * FROM clinical_patients ORDER BY updated_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def delete_patient(self, patient_id: str) -> bool:
        if self.is_postgres:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM clinical_patients WHERE patient_id = %s", (patient_id.strip(),))
                    deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute("DELETE FROM clinical_patients WHERE patient_id = ?", (patient_id.strip(),))
            deleted = cur.rowcount > 0
            self._sqlite_conn.commit()
            return deleted
