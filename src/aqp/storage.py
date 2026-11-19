import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import closing
from .schemas import Evaluation, EvidenceRecord, FailureCode


class StorageManager:
    """
    Quản lý lưu trữ SQLite bất biến cho phiên làm việc, phiên bản bài viết,
    kết quả Evaluation, log trace LLM và xuất dữ liệu JSON/CSV (UC-04, NFR6, NFR7).
    """

    def __init__(self, db_path: str = "aqp_storage.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    product_type TEXT NOT NULL,
                    initial_article TEXT NOT NULL,
                    article_hash TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS article_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    article_version INTEGER NOT NULL,
                    article_text TEXT NOT NULL,
                    article_hash TEXT NOT NULL,
                    edit_ratio REAL,
                    active_review_seconds REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_id, session_id, article_version)
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    article_version INTEGER NOT NULL,
                    rubric_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    run_status TEXT NOT NULL,
                    latency_s REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_traces (
                    trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    evaluation_id TEXT,
                    step_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT,
                    latency_s REAL NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def compute_sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def save_initial_case(
        self,
        case_id: str,
        session_id: str,
        initial_article: str,
        evidence_records: List[EvidenceRecord],
        product_type: str = "fresh_produce",
    ) -> Dict[str, str]:
        """
        Lưu bản đầu vào bất biến (Bước 1-2 Bảng 25, NFR7).
        Nếu case_id đã tồn tại, tuyệt đối không ghi đè dữ liệu bằng chứng gốc.
        """
        article_hash = self.compute_sha256(initial_article)
        evidence_serialized = json.dumps(
            [rec.model_dump() for rec in evidence_records],
            ensure_ascii=False,
            sort_keys=True,
        )
        evidence_hash = self.compute_sha256(evidence_serialized)
        now = datetime.now(timezone.utc).isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO cases (
                    case_id, session_id, product_type,
                    initial_article, article_hash,
                    evidence_json, evidence_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    session_id,
                    product_type,
                    initial_article,
                    article_hash,
                    evidence_serialized,
                    evidence_hash,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO article_versions (
                    case_id, session_id, article_version,
                    article_text, article_hash,
                    edit_ratio, active_review_seconds, created_at
                ) VALUES (?, ?, 1, ?, ?, 0.0, 0.0, ?)
                """,
                (case_id, session_id, initial_article, article_hash, now),
            )

        return {
            "case_id": case_id,
            "session_id": session_id,
            "article_hash": article_hash,
            "evidence_hash": evidence_hash,
        }

    def save_edited_version(
        self,
        case_id: str,
        session_id: str,
        edited_article: str,
        edit_ratio: float,
        active_review_seconds: float,
    ) -> int:
        """
        Lưu văn bản đã sửa thành một article_version mới, giữ nguyên bản cũ (Sequence Diagram Hình 7).
        """
        now = datetime.now(timezone.utc).isoformat()
        article_hash = self.compute_sha256(edited_article)

        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(article_version), 1) AS max_ver
                FROM article_versions
                WHERE case_id = ? AND session_id = ?
                """,
                (case_id, session_id),
            ).fetchone()
            next_version = int(row["max_ver"]) + 1

            conn.execute(
                """
                INSERT INTO article_versions (
                    case_id, session_id, article_version,
                    article_text, article_hash,
                    edit_ratio, active_review_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    session_id,
                    next_version,
                    edited_article,
                    article_hash,
                    edit_ratio,
                    active_review_seconds,
                    now,
                ),
            )
        return next_version

    def save_evaluation(self, eval_data: Evaluation) -> None:
        """Lưu kết quả Evaluation có cấu trúc vào SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        payload_json = eval_data.model_dump_json(indent=2)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations (
                    evaluation_id, case_id, session_id, article_version,
                    rubric_version, model, run_status, latency_s,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_data.evaluation_id,
                    eval_data.case_id,
                    eval_data.session_id,
                    eval_data.article_version,
                    eval_data.rubric_version,
                    eval_data.model,
                    eval_data.run_status.value,
                    eval_data.latency_s,
                    payload_json,
                    now,
                ),
            )

    def load_evaluation(self, evaluation_id: str) -> Optional[Evaluation]:
        """Đọc lại Evaluation từ SQLite (đảm bảo round-trip không mất claim/anchor)."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
            if not row:
                return None
            return Evaluation.model_validate_json(row["payload_json"])

    def log_trace(
        self,
        session_id: str,
        step_name: str,
        model: str,
        latency_s: float,
        evaluation_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Ghi log thời gian phản hồi và lượng token tiêu thụ vào SQLite (thay Langfuse theo Mục 4.2)."""
        now = datetime.now(timezone.utc).isoformat()
        total_tokens = prompt_tokens + completion_tokens
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO llm_traces (
                    session_id, evaluation_id, step_name, model,
                    prompt_version, latency_s, prompt_tokens,
                    completion_tokens, total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    evaluation_id,
                    step_name,
                    model,
                    prompt_version,
                    latency_s,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    now,
                ),
            )

    def export_json(self, eval_data: Evaluation, output_path: Optional[str] = None) -> str:
        """
        Xuất toàn bộ Evaluation ra định dạng JSON chuẩn hóa (Mục 5.7.3 & UC-04),
        bảo toàn cấu trúc lồng nhau của claims, evidence_anchors và suggestion.
        """
        json_str = json.dumps(
            eval_data.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
        return json_str

    def export_csv(self, eval_data: Evaluation, output_path: Optional[str] = None) -> str:
        """
        Xuất nhanh bản ghi tóm tắt ra định dạng CSV (UC-04 Luồng thay thế AC1).
        """
        total_claims = len(eval_data.claims)
        supported = sum(1 for c in eval_data.claims if c.evidence_status.value == "SUPPORTED")
        contradictory = sum(1 for c in eval_data.claims if c.evidence_status.value == "CONTRADICTORY")
        insufficient = sum(1 for c in eval_data.claims if c.evidence_status.value == "INSUFFICIENT_EVIDENCE")
        needs_review = sum(1 for c in eval_data.claims if c.evidence_status.value == "NEEDS_REVIEW")

        all_failures: List[str] = []
        for c in eval_data.claims:
            for code in c.failure_codes:
                val = code.value if isinstance(code, FailureCode) else str(code)
                if val not in all_failures:
                    all_failures.append(val)

        edit_ratio = eval_data.editing.edit_ratio if eval_data.editing else 0.0
        review_sec = eval_data.editing.active_review_seconds if eval_data.editing else 0.0

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "schema_version",
            "case_id",
            "session_id",
            "evaluation_id",
            "article_version",
            "rubric_version",
            "model",
            "run_status",
            "total_claims",
            "supported_claims",
            "contradictory_claims",
            "insufficient_evidence_claims",
            "needs_review_claims",
            "detected_failure_codes",
            "completeness_issues_count",
            "tone_warnings_count",
            "edit_ratio_pct",
            "active_review_seconds",
            "latency_s",
        ])

        completeness_issues = sum(
            1 for item in eval_data.completeness
            if item.status in ("missing_source_has", "missing_source_lacks")
        )

        writer.writerow([
            eval_data.schema_version,
            eval_data.case_id,
            eval_data.session_id,
            eval_data.evaluation_id,
            eval_data.article_version,
            eval_data.rubric_version,
            eval_data.model,
            eval_data.run_status.value,
            total_claims,
            supported,
            contradictory,
            insufficient,
            needs_review,
            ";".join(all_failures),
            completeness_issues,
            len(eval_data.tone_warnings),
            edit_ratio,
            review_sec,
            round(eval_data.latency_s, 3),
        ])

        csv_str = output.getvalue()
        if output_path:
            Path(output_path).write_text(csv_str, encoding="utf-8-sig")
        return csv_str