"""SQLite persistence for completed food-inspection reports.

This module owns the local database connection, uploaded-image storage, report
history, and aggregate metrics. It deliberately uses only Python's built-in
``sqlite3`` package so the application has no separate database dependency.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from backend.schemas import InspectionResult, InspectionStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "runtime_artifacts"
UPLOADS_DIR = ARTIFACTS_DIR / "outputs" / "uploads"
DATABASE_PATH = ARTIFACTS_DIR / "food_inspection.db"


@contextmanager
def _connection() -> Generator[sqlite3.Connection, None, None]:
    """Open one short-lived connection so background jobs remain thread-safe."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """Create the report table and indexes when the API starts."""
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspection_reports (
                id TEXT PRIMARY KEY,
                frame_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                num_detections INTEGER NOT NULL,
                overall_status TEXT NOT NULL,
                average_quality_score REAL,
                image_path TEXT,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_reports_timestamp
                ON inspection_reports(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_reports_status
                ON inspection_reports(overall_status);
            """
        )


def save_uploaded_image(raw_bytes: bytes, original_filename: Optional[str], report_id: str) -> str:
    """Persist the original upload beside the database and return its relative path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = ".jpg"
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_filename or "upload").stem).strip("-")
    filename = f"{report_id}_{safe_stem or 'upload'}{suffix}"
    target = UPLOADS_DIR / filename
    target.write_bytes(raw_bytes)
    return str(target.relative_to(PROJECT_ROOT).as_posix())


def _overall_status(result: InspectionResult) -> InspectionStatus:
    statuses = [item.quality.status for item in result.items]
    if InspectionStatus.DEFECT in statuses:
        return InspectionStatus.DEFECT
    if InspectionStatus.UNCERTAIN in statuses:
        return InspectionStatus.UNCERTAIN
    if not statuses or all(status == InspectionStatus.SKIPPED for status in statuses):
        return InspectionStatus.SKIPPED
    return InspectionStatus.OK


def _average_quality_score(result: InspectionResult) -> Optional[float]:
    scores = [
        item.quality.overall_quality_score
        for item in result.items
        if item.quality.overall_quality_score is not None
    ]
    return sum(scores) / len(scores) if scores else None


def save_inspection(
    result: InspectionResult,
    report_id: str,
    image_path: Optional[str] = None,
) -> InspectionResult:
    """Store a completed report and attach its stable identifier to the API result."""
    initialize_database()
    result.report_id = report_id
    overall_status = _overall_status(result)
    average_quality_score = _average_quality_score(result)
    result_json = result.model_dump_json()

    with _connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO inspection_reports (
                id, frame_id, timestamp, source, image_width, image_height,
                num_detections, overall_status, average_quality_score,
                image_path, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                result.frame_id,
                result.timestamp.isoformat(),
                result.source,
                result.image_size.width,
                result.image_size.height,
                result.num_detections,
                overall_status.value,
                average_quality_score,
                image_path,
                result_json,
            ),
        )
    return result


def _parse_report(row: sqlite3.Row) -> InspectionResult:
    result = InspectionResult.model_validate_json(row["result_json"])
    result.report_id = row["id"]
    return result


def list_reports(
    limit: int = 100,
    offset: int = 0,
    status: Optional[InspectionStatus] = None,
    search: Optional[str] = None,
) -> list[InspectionResult]:
    """Return persisted reports newest first; search matches filename/source."""
    initialize_database()
    clauses: list[str] = []
    values: list[object] = []
    if status is not None:
        clauses.append("overall_status = ?")
        values.append(status.value)
    if search:
        clauses.append("source LIKE ?")
        values.append(f"%{search.strip()}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values.extend([max(1, min(limit, 500)), max(0, offset)])

    with _connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, result_json
            FROM inspection_reports
            {where}
            ORDER BY timestamp DESC, created_at DESC
            LIMIT ? OFFSET ?
            """,
            values,
        ).fetchall()
    return [_parse_report(row) for row in rows]


def get_report(report_id: str) -> Optional[InspectionResult]:
    """Return one persisted report, or ``None`` if it does not exist."""
    initialize_database()
    with _connection() as connection:
        row = connection.execute(
            "SELECT id, result_json FROM inspection_reports WHERE id = ?", (report_id,)
        ).fetchone()
    return _parse_report(row) if row else None


def get_report_summary() -> dict:
    """Derive dashboard metrics exclusively from persisted inspection results."""
    initialize_database()
    with _connection() as connection:
        rows = connection.execute("SELECT id, result_json FROM inspection_reports").fetchall()

    reports = [_parse_report(row) for row in rows]
    items = [item for report in reports for item in report.items]
    status_counts = Counter(item.quality.status.value for item in items)
    label_counts = Counter(item.detection.label for item in items)
    defect_counts = Counter(defect for item in items for defect in item.quality.defects)
    action_counts = Counter(item.quality.required_action.value for item in items)
    quality_scores = [
        item.quality.overall_quality_score
        for item in items
        if item.quality.overall_quality_score is not None
    ]
    confidences = [item.detection.confidence for item in items]

    return {
        "total_inspections": len(reports),
        "total_detections": len(items),
        "ok_count": status_counts[InspectionStatus.OK.value],
        "defect_count": status_counts[InspectionStatus.DEFECT.value],
        "uncertain_count": status_counts[InspectionStatus.UNCERTAIN.value],
        "skipped_count": status_counts[InspectionStatus.SKIPPED.value],
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else None,
        "top_classes": [
            {"label": label, "count": count}
            for label, count in label_counts.most_common(8)
        ],
        "defect_breakdown": [
            {"defect": defect, "count": count}
            for defect, count in defect_counts.most_common(8)
        ],
        "action_counts": {
            "none": action_counts["none"],
            "flag_for_review": action_counts["flag_for_review"],
            "remove": action_counts["remove"],
        },
    }


def export_reports() -> list[dict]:
    """Return the complete persisted report set for JSON export."""
    return [report.model_dump(mode="json") for report in list_reports(limit=500)]
