import re
import json
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
            patient_name TEXT DEFAULT '',
            gender TEXT DEFAULT '',
            age TEXT DEFAULT '',
            weight_kg TEXT DEFAULT '',
            clinical_note TEXT DEFAULT '',
            proposed_treatment TEXT DEFAULT '',
            ai_treatment_evaluation TEXT DEFAULT '',
            history_file_id TEXT DEFAULT '',
            history_filename TEXT DEFAULT '',
            parsed_history_text TEXT DEFAULT '',
            last_xray_filename TEXT DEFAULT '',
            last_xray_prediction TEXT DEFAULT '',
            timeline_json TEXT DEFAULT '[]',
            updated_at TEXT NOT NULL
        );
        """
        new_cols = [
            ("patient_name", "TEXT DEFAULT ''"),
            ("gender", "TEXT DEFAULT ''"),
            ("weight_kg", "TEXT DEFAULT ''"),
            ("proposed_treatment", "TEXT DEFAULT ''"),
            ("ai_treatment_evaluation", "TEXT DEFAULT ''"),
            ("timeline_json", "TEXT DEFAULT '[]'"),
        ]
        if self.is_postgres:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
                    for col_name, col_def in new_cols:
                        cur.execute(
                            f"ALTER TABLE clinical_patients ADD COLUMN IF NOT EXISTS {col_name} {col_def};"
                        )
                    cur.execute(
                        "UPDATE clinical_patients SET last_xray_prediction = '' WHERE last_xray_prediction LIKE '%404%' OR last_xray_prediction LIKE '%Pending%';"
                    )
                conn.commit()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(ddl)
            cur.execute("PRAGMA table_info(clinical_patients);")
            existing_cols = {row[1] for row in cur.fetchall()}
            for col_name, col_def in new_cols:
                if col_name not in existing_cols:
                    cur.execute(f"ALTER TABLE clinical_patients ADD COLUMN {col_name} {col_def};")
            cur.execute(
                "UPDATE clinical_patients SET last_xray_prediction = '' WHERE last_xray_prediction LIKE '%404%' OR last_xray_prediction LIKE '%Pending%';"
            )
            self._sqlite_conn.commit()

    def generate_next_patient_id(self) -> str:
        patients = self.list_patients()
        max_num = 0
        for p in patients:
            pid = p.get("patient_id", "")
            m = re.match(r"^PT-(\d+)$", pid, flags=re.IGNORECASE)
            if m:
                max_num = max(max_num, int(m.group(1)))
        return f"PT-{max_num + 1:04d}"

    def _deserialize(self, row_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not row_dict:
            return None
        out = dict(row_dict)
        pred = out.get("last_xray_prediction") or ""
        if "404" in pred or pred.lower().startswith("pending"):
            out["last_xray_prediction"] = ""
        try:
            raw_tl = json.loads(out.get("timeline_json") or "[]")
            out["timeline"] = [
                t for t in raw_tl
                if "404" not in str(t.get("prediction", ""))
            ]
        except Exception:
            out["timeline"] = []
        return out

    def upsert_patient(
        self,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        gender: Optional[str] = None,
        age: Optional[str] = None,
        weight_kg: Optional[str] = None,
        clinical_note: Optional[str] = None,
        proposed_treatment: Optional[str] = None,
        ai_treatment_evaluation: Optional[str] = None,
        history_file_id: Optional[str] = None,
        history_filename: Optional[str] = None,
        parsed_history_text: Optional[str] = None,
        last_xray_filename: Optional[str] = None,
        last_xray_prediction: Optional[str] = None,
        new_timeline_entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean_id = (patient_id or "").strip()
        if not clean_id or clean_id.upper() == "AUTO":
            clean_id = self.generate_next_patient_id()

        existing = self.get_patient(clean_id) or {}
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        timeline = list(existing.get("timeline") or [])
        if new_timeline_entry:
            entry_with_time = {"timestamp": now_iso, **new_timeline_entry}
            timeline.insert(0, entry_with_time)

        record = {
            "patient_id": clean_id,
            "patient_name": patient_name if patient_name is not None else existing.get("patient_name", ""),
            "gender": gender if gender is not None else existing.get("gender", ""),
            "age": age if age is not None else existing.get("age", ""),
            "weight_kg": weight_kg if weight_kg is not None else existing.get("weight_kg", ""),
            "clinical_note": clinical_note if clinical_note is not None else existing.get("clinical_note", ""),
            "proposed_treatment": proposed_treatment if proposed_treatment is not None else existing.get("proposed_treatment", ""),
            "ai_treatment_evaluation": ai_treatment_evaluation if ai_treatment_evaluation is not None else existing.get("ai_treatment_evaluation", ""),
            "history_file_id": history_file_id if history_file_id is not None else existing.get("history_file_id", ""),
            "history_filename": history_filename if history_filename is not None else existing.get("history_filename", ""),
            "parsed_history_text": parsed_history_text if parsed_history_text is not None else existing.get("parsed_history_text", ""),
            "last_xray_filename": last_xray_filename if last_xray_filename is not None else existing.get("last_xray_filename", ""),
            "last_xray_prediction": last_xray_prediction if last_xray_prediction is not None else existing.get("last_xray_prediction", ""),
            "timeline_json": json.dumps(timeline),
            "updated_at": now_iso,
        }

        if self.is_postgres:
            sql = """
            INSERT INTO clinical_patients (
                patient_id, patient_name, gender, age, weight_kg, clinical_note,
                proposed_treatment, ai_treatment_evaluation, history_file_id, history_filename,
                parsed_history_text, last_xray_filename, last_xray_prediction, timeline_json, updated_at
            ) VALUES (
                %(patient_id)s, %(patient_name)s, %(gender)s, %(age)s, %(weight_kg)s, %(clinical_note)s,
                %(proposed_treatment)s, %(ai_treatment_evaluation)s, %(history_file_id)s, %(history_filename)s,
                %(parsed_history_text)s, %(last_xray_filename)s, %(last_xray_prediction)s, %(timeline_json)s, %(updated_at)s
            )
            ON CONFLICT (patient_id) DO UPDATE SET
                patient_name = EXCLUDED.patient_name,
                gender = EXCLUDED.gender,
                age = EXCLUDED.age,
                weight_kg = EXCLUDED.weight_kg,
                clinical_note = EXCLUDED.clinical_note,
                proposed_treatment = EXCLUDED.proposed_treatment,
                ai_treatment_evaluation = EXCLUDED.ai_treatment_evaluation,
                history_file_id = EXCLUDED.history_file_id,
                history_filename = EXCLUDED.history_filename,
                parsed_history_text = EXCLUDED.parsed_history_text,
                last_xray_filename = EXCLUDED.last_xray_filename,
                last_xray_prediction = EXCLUDED.last_xray_prediction,
                timeline_json = EXCLUDED.timeline_json,
                updated_at = EXCLUDED.updated_at;
            """
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, record)
                conn.commit()
        else:
            sql = """
            INSERT OR REPLACE INTO clinical_patients (
                patient_id, patient_name, gender, age, weight_kg, clinical_note,
                proposed_treatment, ai_treatment_evaluation, history_file_id, history_filename,
                parsed_history_text, last_xray_filename, last_xray_prediction, timeline_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cur = self._sqlite_conn.cursor()
            cur.execute(
                sql,
                (
                    record["patient_id"],
                    record["patient_name"],
                    record["gender"],
                    record["age"],
                    record["weight_kg"],
                    record["clinical_note"],
                    record["proposed_treatment"],
                    record["ai_treatment_evaluation"],
                    record["history_file_id"],
                    record["history_filename"],
                    record["parsed_history_text"],
                    record["last_xray_filename"],
                    record["last_xray_prediction"],
                    record["timeline_json"],
                    record["updated_at"],
                ),
            )
            self._sqlite_conn.commit()

        return self._deserialize(record)

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if self.is_postgres:
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM clinical_patients WHERE patient_id = %s", (patient_id.strip(),))
                    row = cur.fetchone()
                    return self._deserialize(dict(row)) if row else None
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT * FROM clinical_patients WHERE patient_id = ?", (patient_id.strip(),))
            row = cur.fetchone()
            return self._deserialize(dict(row)) if row else None

    def search_patients(self, query: str = "") -> List[Dict[str, Any]]:
        all_patients = self.list_patients()
        q = query.strip().lower()
        if not q:
            return all_patients
        return [
            p for p in all_patients
            if q in p.get("patient_id", "").lower() or q in p.get("patient_name", "").lower()
        ]

    def list_patients(self) -> List[Dict[str, Any]]:
        if self.is_postgres:
            with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM clinical_patients ORDER BY updated_at DESC")
                    return [self._deserialize(dict(r)) for r in cur.fetchall()]
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT * FROM clinical_patients ORDER BY updated_at DESC")
            return [self._deserialize(dict(r)) for r in cur.fetchall()]

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
            cur.execute("SELECT * FROM clinical_patients WHERE patient_id = ?", (patient_id.strip(),))
            deleted = cur.rowcount > 0
            self._sqlite_conn.commit()
            return deleted
