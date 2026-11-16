"""
Forensic Audit Trail & SQLite History Database Module.

Provides persistent SQLite storage for image and video forensic examinations,
recording separate Case IDs and Evidence IDs, exact cryptographic SHA-256 digests
of raw uploaded media bytes (distinct from aligned-crop hashes), media dimensions,
calibration thresholds, processing status, and an immutable append-only `audit_events`
lifecycle ledger (`upload`, `processing`, `prediction`, `report_generation`, `archive_action`).

Important Forensic Scope Notice:
  This module implements local **software forensic audit logs** rather than a
  standalone legally admissible chain of custody (see `CHAIN_OF_CUSTODY_DISCLAIMER`).
"""

from datetime import datetime, timezone
import hashlib
import io
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from PIL import Image


EVENT_UPLOAD = "upload"
EVENT_PROCESSING = "processing"
EVENT_PREDICTION = "prediction"
EVENT_REPORT_GENERATION = "report_generation"
EVENT_ARCHIVE_ACTION = "archive_action"

LIFECYCLE_EVENT_TYPES: Tuple[str, ...] = (
    EVENT_UPLOAD,
    EVENT_PROCESSING,
    EVENT_PREDICTION,
    EVENT_REPORT_GENERATION,
    EVENT_ARCHIVE_ACTION,
)

CHAIN_OF_CUSTODY_DISCLAIMER = (
    "Software Audit Log vs. Legal Chain of Custody: Records stored in this SQLite ledger "
    "and its append-only `audit_events` table constitute application-level software forensic "
    "audit logs (local SHA-256 file digests, workstation UTC timestamps, model configuration, "
    "and calibrated inference metadata) intended for reproducibility and post-upload tamper checks. "
    "They do NOT independently establish a legally admissible chain of custody, which additionally "
    "requires RFC 3161 external trusted timestamping authorities (TSA), hardware write-blockers "
    "or WORM media, PKI/HSM cryptographic signatures, original capture device provenance, and "
    "formally attested custody transfers."
)

# Ordered non-destructive migration definitions for legacy `forensic_examinations` tables
PHASE8_MIGRATION_COLUMNS: List[Tuple[str, str]] = [
    ("evidence_id", "TEXT DEFAULT ''"),
    ("media_type", "TEXT NOT NULL DEFAULT 'image'"),
    ("raw_file_sha256", "TEXT NOT NULL DEFAULT 'N/A'"),
    ("file_size_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("media_dimensions", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("decision_threshold", "REAL NOT NULL DEFAULT 0.5"),
    ("calibrated_temperature", "REAL NOT NULL DEFAULT 1.0"),
    ("prediction_outcome", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
    ("processing_status", "TEXT NOT NULL DEFAULT 'completed'"),
    ("is_archived", "INTEGER NOT NULL DEFAULT 0"),
    ("metadata_json", "TEXT"),
]


def _utc_now_str() -> str:
    """Returns a sortable millisecond-resolution UTC timestamp string."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d} UTC"


def generate_case_id() -> str:
    """Generates a unique forensic examination Case ID (`DFS-...`)."""
    return f"DFS-{int(time.time() * 1000)}-{uuid.uuid4().hex[:4].upper()}"


def generate_evidence_id() -> str:
    """Generates a unique digital Evidence artifact ID (`EVD-...`)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"EVD-{stamp}-{uuid.uuid4().hex[:6].upper()}"


def generate_event_id() -> str:
    """Generates a unique immutable audit event ID (`EVT-...`)."""
    return f"EVT-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6].upper()}"


def compute_bytes_sha256(raw_bytes: bytes) -> str:
    """Computes exact cryptographic SHA-256 digest of raw binary payload bytes."""
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Computes streaming SHA-256 digest of raw file bytes on disk."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_stream_sha256(stream: Any, chunk_size: int = 1024 * 1024) -> Tuple[str, int]:
    """
    Computes streaming SHA-256 digest and byte count from a binary file-like stream,
    resetting position to 0 before and after reading when seekable.
    """
    h = hashlib.sha256()
    total_bytes = 0
    if hasattr(stream, "seek"):
        stream.seek(0)
    try:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            total_bytes += len(chunk)
    finally:
        if hasattr(stream, "seek"):
            stream.seek(0)
    return h.hexdigest(), total_bytes


def compute_image_sha256(pil_image: Image.Image) -> str:
    """Computes SHA-256 digest of encoded PIL Image PNG bytes (post-crop/processed view)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


class ForensicDatabase:
    CHAIN_OF_CUSTODY_DISCLAIMER = CHAIN_OF_CUSTODY_DISCLAIMER

    def __init__(self, db_path: Optional[str] = None):
        env_db = os.environ.get("DFAKE_DB_PATH")
        if db_path is None and env_db:
            db_path = env_db

        if db_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            data_dir = repo_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "forensic_history.db")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path)

        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """
        Initializes upgraded forensic tables, applies non-destructive `ALTER TABLE`
        schema migrations on legacy databases (preserving all existing rows), and creates
        the append-only `audit_events` ledger with SQLite immutability triggers.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Main examinations ledger (fresh install uses full Phase 8 schema)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forensic_examinations (
                    case_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    verdict_tier TEXT NOT NULL,
                    prediction_label TEXT NOT NULL,
                    fake_probability REAL NOT NULL,
                    confidence_percent REAL NOT NULL,
                    neural_score REAL NOT NULL,
                    spectral_score REAL NOT NULL,
                    ela_score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    dominant_region TEXT,
                    has_deconv_spikes INTEGER,
                    device TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    evidence_id TEXT DEFAULT '',
                    media_type TEXT NOT NULL DEFAULT 'image',
                    raw_file_sha256 TEXT NOT NULL DEFAULT 'N/A',
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    media_dimensions TEXT NOT NULL DEFAULT 'unknown',
                    decision_threshold REAL NOT NULL DEFAULT 0.5,
                    calibrated_temperature REAL NOT NULL DEFAULT 1.0,
                    prediction_outcome TEXT NOT NULL DEFAULT 'UNKNOWN',
                    processing_status TEXT NOT NULL DEFAULT 'completed',
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT
                )
                """
            )

            # 2. Non-destructive schema migration for pre-Phase 8 databases
            cursor.execute("PRAGMA table_info(forensic_examinations)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            for col_name, col_def in PHASE8_MIGRATION_COLUMNS:
                if col_name not in existing_cols:
                    cursor.execute(
                        f"ALTER TABLE forensic_examinations ADD COLUMN {col_name} {col_def}"
                    )

            # Assign deterministic Evidence IDs to legacy migrated rows without altering any original columns
            cursor.execute(
                """
                UPDATE forensic_examinations
                SET evidence_id = 'EVD-LEGACY-' || substr(case_id, 1, 24)
                WHERE evidence_id IS NULL OR evidence_id = ''
                """
            )

            # 3. Immutable append-only audit lifecycle table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    case_id TEXT,
                    event_type TEXT NOT NULL,
                    event_timestamp_utc TEXT NOT NULL,
                    sha256_at_event TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'forensic_sentinel_app',
                    details_json TEXT
                )
                """
            )

            # Database-level immutability triggers preventing UPDATE or DELETE on audit_events
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_update
                BEFORE UPDATE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit_events table is append-only and immutable; UPDATE prohibited.');
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_delete
                BEFORE DELETE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit_events table is append-only and immutable; DELETE prohibited.');
                END
                """
            )

            # 4. Indexes for forensic lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON forensic_examinations(timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_id ON forensic_examinations(evidence_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_type ON forensic_examinations(media_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_is_archived ON forensic_examinations(is_archived)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_evidence ON audit_events(evidence_id, event_timestamp_utc ASC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_events(case_id, event_timestamp_utc ASC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)"
            )
            conn.commit()

    def get_schema_info(self) -> Dict[str, Any]:
        """Returns introspected table columns, triggers, and row counts for verification."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(forensic_examinations)")
            exam_cols = [row["name"] for row in cursor.fetchall()]
            cursor.execute("PRAGMA table_info(audit_events)")
            audit_cols = [row["name"] for row in cursor.fetchall()]
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name")
            triggers = [row["name"] for row in cursor.fetchall()]
            exam_total = cursor.execute("SELECT COUNT(*) FROM forensic_examinations").fetchone()[0]
            exam_active = cursor.execute(
                "SELECT COUNT(*) FROM forensic_examinations WHERE is_archived = 0"
            ).fetchone()[0]
            audit_total = cursor.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

        return {
            "db_path": self.db_path,
            "forensic_examinations_columns": exam_cols,
            "audit_events_columns": audit_cols,
            "immutability_triggers": triggers,
            "total_examinations": int(exam_total),
            "active_examinations": int(exam_active),
            "archived_examinations": int(exam_total - exam_active),
            "total_audit_events": int(audit_total),
        }

    def log_audit_event(
        self,
        evidence_id: str,
        event_type: str,
        sha256_at_event: str = "N/A",
        case_id: Optional[str] = None,
        actor: str = "forensic_sentinel_app",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Appends a single immutable lifecycle event into `audit_events`."""
        if event_type not in LIFECYCLE_EVENT_TYPES:
            raise ValueError(
                f"Unsupported audit event_type '{event_type}'. Allowed: {LIFECYCLE_EVENT_TYPES}"
            )
        if not evidence_id:
            raise ValueError("evidence_id must be non-empty when logging an audit event.")

        event_id = generate_event_id()
        ts = _utc_now_str()
        details_payload = json.dumps(details or {}, sort_keys=True, default=str)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    event_id, evidence_id, case_id, event_type,
                    event_timestamp_utc, sha256_at_event, actor, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    evidence_id,
                    case_id,
                    event_type,
                    ts,
                    sha256_at_event or "N/A",
                    actor,
                    details_payload,
                ),
            )
            conn.commit()
        return event_id

    def get_audit_events(
        self,
        evidence_id: Optional[str] = None,
        case_id: Optional[str] = None,
        event_type: Optional[str] = None,
        descending: bool = False,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Retrieves immutable lifecycle events filtered by Evidence ID, Case ID, or event type."""
        clauses = []
        params: List[Any] = []
        if evidence_id:
            clauses.append("evidence_id = ?")
            params.append(evidence_id)
        if case_id:
            clauses.append("case_id = ?")
            params.append(case_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_dir = "DESC" if descending else "ASC"
        params.append(int(limit))

        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_events {where_sql} ORDER BY event_timestamp_utc {order_dir}, rowid {order_dir} LIMIT ?",
                params,
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            rec = dict(r)
            try:
                rec["details"] = json.loads(rec.get("details_json") or "{}")
            except Exception:
                rec["details"] = {}
            out.append(rec)
        return out

    def record_media_upload(
        self,
        filename: str,
        raw_file_bytes: Optional[bytes] = None,
        raw_file_path: Optional[Union[str, Path]] = None,
        raw_file_sha256: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        media_type: str = "image",
        media_dimensions: str = "unknown",
        evidence_id: Optional[str] = None,
        actor: str = "streamlit_ui",
    ) -> Dict[str, Any]:
        """
        Records initial raw-media intake (`upload` lifecycle event) with exact
        cryptographic SHA-256 of raw uploaded bytes prior to any crop or processing.
        """
        resolved_evd = evidence_id or generate_evidence_id()
        resolved_sha = raw_file_sha256 or "N/A"
        resolved_size = int(file_size_bytes) if file_size_bytes is not None else 0

        if raw_file_bytes is not None:
            resolved_sha = compute_bytes_sha256(raw_file_bytes)
            if file_size_bytes is None:
                resolved_size = len(raw_file_bytes)
        elif raw_file_path is not None and Path(raw_file_path).exists():
            p = Path(raw_file_path)
            resolved_sha = compute_file_sha256(p)
            if file_size_bytes is None:
                resolved_size = int(p.stat().st_size)

        event_id = self.log_audit_event(
            evidence_id=resolved_evd,
            event_type=EVENT_UPLOAD,
            sha256_at_event=resolved_sha,
            case_id=None,
            actor=actor,
            details={
                "filename": filename,
                "media_type": media_type,
                "file_size_bytes": resolved_size,
                "media_dimensions": media_dimensions,
            },
        )
        return {
            "evidence_id": resolved_evd,
            "event_id": event_id,
            "raw_file_sha256": resolved_sha,
            "file_size_bytes": resolved_size,
            "media_dimensions": media_dimensions,
        }

    def log_examination(
        self,
        results: Dict[str, Any],
        image_source_name: str = "",
        device_str: str = "MPS",
        model_version: str = "DualDomainSentinel-v1.0",
        *,
        raw_file_bytes: Optional[bytes] = None,
        raw_file_path: Optional[Union[str, Path]] = None,
        raw_file_sha256: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        media_type: Optional[str] = None,
        media_dimensions: Optional[str] = None,
        evidence_id: Optional[str] = None,
        case_id: Optional[str] = None,
        processing_status: Optional[str] = None,
        log_lifecycle_events: bool = True,
        upload_already_logged: bool = False,
        actor: str = "forensic_sentinel_app",
    ) -> str:
        """
        Logs an image or video forensic examination into `forensic_examinations`
        and appends corresponding lifecycle events (`upload`, `processing`, `prediction`)
        into the immutable `audit_events` table.

        Returns the generated Case ID (`DFS-...`).
        """
        resolved_case_id = case_id or results.get("case_id") or generate_case_id()
        resolved_evidence_id = evidence_id or results.get("evidence_id") or generate_evidence_id()
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        # Determine media type
        if media_type is not None:
            resolved_media_type = str(media_type).lower().strip()
        elif any(k in results for k in ("timeline", "duration_sec", "overall_assessment", "analysis_scope")):
            resolved_media_type = "video"
        else:
            resolved_media_type = "image"

        resolved_name = image_source_name or results.get("filename") or f"Unnamed_{resolved_media_type}"
        file_meta = results.get("file_metadata", {}) if isinstance(results.get("file_metadata"), dict) else {}

        # Resolve exact raw uploaded file SHA-256 and byte length (B9 fix: never conflated with post-crop PNG hash)
        resolved_raw_sha = raw_file_sha256 or results.get("raw_file_sha256") or results.get("sha256") or "N/A"
        resolved_size = file_size_bytes if file_size_bytes is not None else (
            results.get("file_size_bytes") or file_meta.get("file_size_bytes") or 0
        )
        if raw_file_bytes is not None:
            resolved_raw_sha = compute_bytes_sha256(raw_file_bytes)
            if file_size_bytes is None:
                resolved_size = len(raw_file_bytes)
        elif raw_file_path is not None and Path(raw_file_path).exists():
            p = Path(raw_file_path)
            resolved_raw_sha = compute_file_sha256(p)
            if file_size_bytes is None:
                resolved_size = int(p.stat().st_size)
        resolved_size = int(resolved_size or 0)

        # Compute aligned-crop PNG SHA-256 separately for image evidence
        if "aligned_face" in results and isinstance(results["aligned_face"], Image.Image):
            crop_sha256 = compute_image_sha256(results["aligned_face"])
        else:
            crop_sha256 = "N/A"

        # Resolve media dimensions string
        if media_dimensions:
            resolved_dims = str(media_dimensions)
        elif results.get("media_dimensions"):
            resolved_dims = str(results["media_dimensions"])
        elif resolved_media_type == "video":
            w = int(file_meta.get("width", 0) or 0)
            h = int(file_meta.get("height", 0) or 0)
            fps = float(results.get("fps", file_meta.get("fps", 0.0)) or 0.0)
            dur = float(results.get("duration_sec", file_meta.get("duration_sec", 0.0)) or 0.0)
            fc = int(results.get("frame_count", file_meta.get("frame_count", 0)) or 0)
            resolved_dims = f"{w}x{h} @ {fps:.2f}fps, {dur:.2f}s ({fc} frames)"
        elif isinstance(results.get("full_image_annotated"), Image.Image):
            full_img = results["full_image_annotated"]
            resolved_dims = f"{full_img.width}x{full_img.height}"
        elif isinstance(results.get("aligned_face"), Image.Image):
            af = results["aligned_face"]
            resolved_dims = f"{af.width}x{af.height} (aligned_crop)"
        else:
            resolved_dims = "unknown"

        cal_meta = results.get("calibration_metadata", {}) if isinstance(results.get("calibration_metadata"), dict) else {}
        decision_threshold = float(cal_meta.get("decision_threshold", 0.50))
        calibrated_temperature = float(cal_meta.get("temperature", 1.0))

        if resolved_media_type == "video":
            resolved_status = str(processing_status or results.get("status") or "completed")
            verdict_tier = str(results.get("overall_assessment", results.get("prediction_outcome", "Unknown")))
            prediction_label = verdict_tier
            prediction_outcome = str(results.get("prediction_outcome", verdict_tier))
            agg_p = results.get("aggregated_fake_probability")
            fake_prob = float(agg_p) if agg_p is not None else 0.5
            conf_pct = float(results.get("aggregated_confidence_percent", 0.0))
            neural_score = float(agg_p) if agg_p is not None else 0.5
            spectral_score = 0.0
            ela_score = 0.0
            dominant_region = "Frame-Level Video Timeline"
            has_deconv_spikes = 0
            metadata_extra = {
                "processed_frames": int(results.get("processed_frames", 0) or 0),
                "detected_faces": int(results.get("detected_faces", 0) or 0),
                "frames_with_faces": int(results.get("frames_with_faces", 0) or 0),
                "authentic_frame_count": int(results.get("authentic_frame_count", 0) or 0),
                "manipulated_frame_count": int(results.get("manipulated_frame_count", 0) or 0),
                "uncertain_frame_count": int(results.get("uncertain_frame_count", 0) or 0),
                "suspicious_intervals_count": len(results.get("suspicious_timestamps", [])),
                "analysis_scope": results.get("analysis_scope", "frame_level_inference"),
                "temporal_modeling": bool(results.get("temporal_modeling", False)),
                "sampling_config": results.get("sampling_config", {}),
                "auxiliary_scores_applicable": False,
                "error": results.get("error"),
            }
        else:
            resolved_status = str(processing_status or results.get("status") or "completed")
            verdict_tier = str(results.get("verdict_tier", "Unknown"))
            prediction_label = str(results.get("prediction_label", "Unknown"))
            prediction_outcome = str(results.get("prediction_outcome", verdict_tier))
            fake_prob = float(results.get("fake_probability", 0.5))
            conf_pct = float(results.get("confidence_percent", 50.0))
            ev = results.get("forensic_evidence", {}) if isinstance(results.get("forensic_evidence"), dict) else {}
            spec = results.get("spectral_details", {}) if isinstance(results.get("spectral_details"), dict) else {}
            landmarks = results.get("spatial_landmarks", {}) if isinstance(results.get("spatial_landmarks"), dict) else {}
            neural_score = float(ev.get("neural_fake_prob", fake_prob))
            spectral_score = float(ev.get("spectral_anomaly_prob", 0.5))
            ela_score = float(ev.get("ela_anomaly_prob", 0.5))
            dominant_region = str(landmarks.get("dominant_region", "N/A"))
            has_deconv_spikes = 1 if spec.get("has_deconv_spikes", False) else 0
            metadata_extra = {
                "faces_count": int(results.get("faces_count", 1 if results.get("face_detected", False) else 0) or 0),
                "face_detected": bool(results.get("face_detected", False)),
                "uncertainty_reason": results.get("uncertainty_reason"),
                "calibrated": bool(cal_meta.get("calibrated", False)),
                "uncertain_low": cal_meta.get("uncertain_low"),
                "uncertain_high": cal_meta.get("uncertain_high"),
                "crop_sha256": crop_sha256,
            }

        latency_ms = float(results.get("inference_time_ms", 0.0) or 0.0)
        metadata_json = json.dumps(metadata_extra, sort_keys=True, default=str)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO forensic_examinations (
                    case_id, timestamp, image_name, image_sha256,
                    verdict_tier, prediction_label, fake_probability, confidence_percent,
                    neural_score, spectral_score, ela_score, latency_ms,
                    dominant_region, has_deconv_spikes, device, model_version,
                    evidence_id, media_type, raw_file_sha256, file_size_bytes,
                    media_dimensions, decision_threshold, calibrated_temperature,
                    prediction_outcome, processing_status, is_archived, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_case_id,
                    timestamp_str,
                    resolved_name,
                    crop_sha256,
                    verdict_tier,
                    prediction_label,
                    fake_prob,
                    conf_pct,
                    neural_score,
                    spectral_score,
                    ela_score,
                    latency_ms,
                    dominant_region,
                    has_deconv_spikes,
                    device_str,
                    model_version,
                    resolved_evidence_id,
                    resolved_media_type,
                    resolved_raw_sha,
                    resolved_size,
                    resolved_dims,
                    decision_threshold,
                    calibrated_temperature,
                    prediction_outcome,
                    resolved_status,
                    0,
                    metadata_json,
                ),
            )
            conn.commit()

        # Propagate IDs back onto caller dictionary for UI/report display
        if isinstance(results, dict):
            results["case_id"] = resolved_case_id
            results["evidence_id"] = resolved_evidence_id
            results["raw_file_sha256"] = resolved_raw_sha
            results["file_size_bytes"] = resolved_size
            results["media_dimensions"] = resolved_dims

        if log_lifecycle_events:
            if not upload_already_logged:
                self.log_audit_event(
                    evidence_id=resolved_evidence_id,
                    case_id=resolved_case_id,
                    event_type=EVENT_UPLOAD,
                    sha256_at_event=resolved_raw_sha,
                    actor=actor,
                    details={
                        "filename": resolved_name,
                        "media_type": resolved_media_type,
                        "file_size_bytes": resolved_size,
                        "media_dimensions": resolved_dims,
                    },
                )
            self.log_audit_event(
                evidence_id=resolved_evidence_id,
                case_id=resolved_case_id,
                event_type=EVENT_PROCESSING,
                sha256_at_event=resolved_raw_sha,
                actor=actor,
                details={
                    "processing_status": resolved_status,
                    "device": device_str,
                    "model_version": model_version,
                    "latency_ms": round(latency_ms, 2),
                    "crop_sha256": crop_sha256,
                    "media_type": resolved_media_type,
                },
            )
            self.log_audit_event(
                evidence_id=resolved_evidence_id,
                case_id=resolved_case_id,
                event_type=EVENT_PREDICTION,
                sha256_at_event=resolved_raw_sha,
                actor=actor,
                details={
                    "prediction_outcome": prediction_outcome,
                    "verdict_tier": verdict_tier,
                    "fake_probability": round(fake_prob, 5),
                    "confidence_percent": round(conf_pct, 2),
                    "decision_threshold": round(decision_threshold, 4),
                    "calibrated_temperature": round(calibrated_temperature, 4),
                },
            )

        return resolved_case_id

    def log_video_examination(
        self,
        video_results: Dict[str, Any],
        device_str: str = "CPU",
        model_version: str = "DualDomainSentinel-v1.0",
        *,
        raw_file_bytes: Optional[bytes] = None,
        raw_file_path: Optional[Union[str, Path]] = None,
        evidence_id: Optional[str] = None,
        case_id: Optional[str] = None,
        upload_already_logged: bool = False,
        actor: str = "forensic_sentinel_app",
    ) -> str:
        """Convenience wrapper for logging a frame-level video forensic examination."""
        return self.log_examination(
            video_results,
            image_source_name=str(video_results.get("filename") or "Uploaded_Video"),
            device_str=device_str,
            model_version=model_version,
            raw_file_bytes=raw_file_bytes,
            raw_file_path=raw_file_path,
            media_type="video",
            evidence_id=evidence_id,
            case_id=case_id,
            upload_already_logged=upload_already_logged,
            actor=actor,
        )

    def record_report_generation(
        self,
        case_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        report_format: str = "pdf",
        report_bytes: Optional[bytes] = None,
        output_name: Optional[str] = None,
        actor: str = "streamlit_ui",
    ) -> str:
        """Logs an immutable `report_generation` event whenever an official report is compiled/exported."""
        resolved_evd = evidence_id
        raw_sha = "N/A"
        if case_id:
            rec = self.get_record(case_id)
            if rec is not None:
                resolved_evd = resolved_evd or rec.get("evidence_id")
                raw_sha = rec.get("raw_file_sha256") or "N/A"

        resolved_evd = resolved_evd or generate_evidence_id()
        report_sha = compute_bytes_sha256(report_bytes) if report_bytes is not None else None
        return self.log_audit_event(
            evidence_id=resolved_evd,
            case_id=case_id,
            event_type=EVENT_REPORT_GENERATION,
            sha256_at_event=raw_sha,
            actor=actor,
            details={
                "report_format": report_format,
                "output_name": output_name,
                "report_sha256": report_sha,
                "report_size_bytes": len(report_bytes) if report_bytes is not None else None,
            },
        )

    def get_record(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single examination record by `case_id`."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM forensic_examinations WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        rec = dict(row)
        try:
            rec["metadata"] = json.loads(rec.get("metadata_json") or "{}")
        except Exception:
            rec["metadata"] = {}
        return rec

    def get_history(
        self,
        limit: int = 100,
        include_archived: bool = False,
        media_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves forensic examination records sorted newest-first."""
        clauses = []
        params: List[Any] = []
        if not include_archived:
            clauses.append("COALESCE(is_archived, 0) = 0")
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type.lower().strip())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))

        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM forensic_examinations {where_sql} ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                params,
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            rec = dict(r)
            try:
                rec["metadata"] = json.loads(rec.get("metadata_json") or "{}")
            except Exception:
                rec["metadata"] = {}
            results.append(rec)
        return results

    def verify_evidence_integrity(
        self,
        case_id: str,
        raw_bytes: Optional[bytes] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Re-hashes candidate raw media bytes or file against the stored `raw_file_sha256`
        of `case_id` to detect post-upload file modification or truncation.
        """
        rec = self.get_record(case_id)
        if rec is None:
            raise KeyError(f"Case ID '{case_id}' not found in forensic_examinations.")

        stored_sha = rec.get("raw_file_sha256") or "N/A"
        if raw_bytes is not None:
            candidate_sha = compute_bytes_sha256(raw_bytes)
            candidate_size = len(raw_bytes)
        elif file_path is not None and Path(file_path).exists():
            p = Path(file_path)
            candidate_sha = compute_file_sha256(p)
            candidate_size = int(p.stat().st_size)
        else:
            raise ValueError("Provide either `raw_bytes` or an existing `file_path` to verify.")

        is_valid_digest = len(stored_sha) == 64
        matches = bool(is_valid_digest and candidate_sha == stored_sha)
        return {
            "case_id": case_id,
            "evidence_id": rec.get("evidence_id"),
            "stored_raw_file_sha256": stored_sha,
            "candidate_sha256": candidate_sha,
            "stored_file_size_bytes": int(rec.get("file_size_bytes", 0) or 0),
            "candidate_size_bytes": candidate_size,
            "integrity_verified": matches,
            "status": "MATCH" if matches else ("NO_STORED_RAW_HASH" if not is_valid_digest else "MISMATCH_TAMPERED"),
        }

    def archive_history(
        self,
        confirm: bool = False,
        actor: str = "operator",
        reason: str = "Operator confirmed archiving active examination records",
    ) -> int:
        """
        Non-destructively archives all active records (`is_archived = 1`) and appends
        an immutable `archive_action` audit event. Requires explicit `confirm=True`.
        """
        return self.clear_history(confirm=confirm, archive_only=True, actor=actor, reason=reason)

    def clear_history(
        self,
        confirm: bool = False,
        archive_only: bool = True,
        actor: str = "operator",
        reason: str = "Operator confirmed clearing active examination view",
    ) -> int:
        """
        Safely archives (default) or deletes active `forensic_examinations` entries ONLY when
        `confirm=True` is explicitly provided. Immutable `audit_events` are NEVER deleted.
        """
        if not confirm:
            raise ValueError(
                "Explicit confirmation (confirm=True) is required before clearing or archiving examination history."
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if archive_only:
                active_count = cursor.execute(
                    "SELECT COUNT(*) FROM forensic_examinations WHERE COALESCE(is_archived, 0) = 0"
                ).fetchone()[0]
                cursor.execute(
                    """
                    UPDATE forensic_examinations
                    SET is_archived = 1
                    WHERE COALESCE(is_archived, 0) = 0
                    """
                )
            else:
                active_count = cursor.execute("SELECT COUNT(*) FROM forensic_examinations").fetchone()[0]
                cursor.execute("DELETE FROM forensic_examinations")
            conn.commit()

        self.log_audit_event(
            evidence_id="SYSTEM-LEDGER",
            case_id=None,
            event_type=EVENT_ARCHIVE_ACTION,
            sha256_at_event="N/A",
            actor=actor,
            details={
                "action": "archive_active_records" if archive_only else "delete_examination_rows",
                "archive_only": bool(archive_only),
                "affected_records": int(active_count),
                "reason": reason,
            },
        )
        return int(active_count)

    def get_summary_stats(self, include_archived: bool = False) -> Dict[str, Any]:
        """Returns aggregate statistics across stored forensic examinations."""
        records = self.get_history(limit=10000, include_archived=include_archived)
        total = len(records)
        manip = sum(1 for r in records if str(r.get("prediction_outcome", r.get("verdict_tier", ""))).upper().startswith("MANIPULATED") or "manipulated" in str(r.get("verdict_tier", "")).lower())
        auth = sum(1 for r in records if str(r.get("prediction_outcome", r.get("verdict_tier", ""))).upper().startswith("AUTHENTIC") or "authentic" in str(r.get("verdict_tier", "")).lower())
        uncert = max(0, total - manip - auth)
        imgs = sum(1 for r in records if r.get("media_type", "image") == "image")
        vids = sum(1 for r in records if r.get("media_type", "image") == "video")
        return {
            "total": total,
            "manipulated": manip,
            "authentic": auth,
            "uncertain": uncert,
            "images": imgs,
            "videos": vids,
        }
