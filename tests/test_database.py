"""
Comprehensive unit and migration tests for Phase 8 ForensicDatabase upgrades.
"""

import io
import sqlite3
import tempfile
import pytest
from PIL import Image

from src.core.database import (
    CHAIN_OF_CUSTODY_DISCLAIMER,
    EVENT_ARCHIVE_ACTION,
    EVENT_PREDICTION,
    EVENT_PROCESSING,
    EVENT_REPORT_GENERATION,
    EVENT_UPLOAD,
    ForensicDatabase,
    compute_bytes_sha256,
    compute_image_sha256,
    compute_stream_sha256,
)


def _make_jpeg_bytes(color=(255, 0, 0), size=(320, 240)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_raw_bytes_hash_vs_processed_crop_and_tamper_detection():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = ForensicDatabase(db_path=tmp.name)

        raw_bytes = _make_jpeg_bytes(color=(120, 180, 210), size=(400, 300))
        crop_img = Image.new("RGB", (256, 256), color=(120, 180, 210))

        expected_raw_sha = compute_bytes_sha256(raw_bytes)
        expected_crop_sha = compute_image_sha256(crop_img)
        assert len(expected_raw_sha) == 64
        assert len(expected_crop_sha) == 64
        assert expected_raw_sha != expected_crop_sha

        stream_sha, stream_len = compute_stream_sha256(io.BytesIO(raw_bytes))
        assert stream_sha == expected_raw_sha
        assert stream_len == len(raw_bytes)

        intake = db.record_media_upload(
            filename="suspect_raw.jpg",
            raw_file_bytes=raw_bytes,
            media_type="image",
            media_dimensions="400x300",
        )
        assert intake["raw_file_sha256"] == expected_raw_sha
        assert intake["file_size_bytes"] == len(raw_bytes)
        assert intake["evidence_id"].startswith("EVD-")

        mock_results = {
            "aligned_face": crop_img,
            "verdict_tier": "MANIPULATED",
            "prediction_outcome": "MANIPULATED",
            "prediction_label": "MANIPULATED",
            "fake_probability": 0.87,
            "confidence_percent": 87.0,
            "inference_time_ms": 62.5,
            "calibration_metadata": {
                "calibrated": True,
                "temperature": 2.6543,
                "decision_threshold": 0.71,
                "uncertain_low": 0.707,
                "uncertain_high": 0.7188,
            },
            "forensic_evidence": {
                "neural_fake_prob": 0.87,
                "spectral_anomaly_prob": 0.76,
                "ela_anomaly_prob": 0.64,
            },
            "spectral_details": {"has_deconv_spikes": True},
            "spatial_landmarks": {"dominant_region": "Upper / Eye & Brow Region"},
        }

        case_id = db.log_examination(
            mock_results,
            image_source_name="suspect_raw.jpg",
            device_str="CPU",
            raw_file_bytes=raw_bytes,
            media_type="image",
            media_dimensions="400x300",
            evidence_id=intake["evidence_id"],
            upload_already_logged=True,
        )
        assert case_id.startswith("DFS-")

        rec = db.get_record(case_id)
        assert rec is not None
        assert rec["evidence_id"] == intake["evidence_id"]
        assert rec["media_type"] == "image"
        assert rec["raw_file_sha256"] == expected_raw_sha
        assert rec["image_sha256"] == expected_crop_sha
        assert rec["file_size_bytes"] == len(raw_bytes)
        assert rec["media_dimensions"] == "400x300"
        assert rec["decision_threshold"] == pytest.approx(0.71)
        assert rec["calibrated_temperature"] == pytest.approx(2.6543)
        assert rec["prediction_outcome"] == "MANIPULATED"
        assert rec["processing_status"] == "completed"

        # Exact raw-bytes verification passes
        ok_check = db.verify_evidence_integrity(case_id, raw_bytes=raw_bytes)
        assert ok_check["integrity_verified"] is True
        assert ok_check["status"] == "MATCH"

        # 1-byte modification is caught immediately
        tampered = bytearray(raw_bytes)
        tampered[-1] ^= 0xFF
        bad_check = db.verify_evidence_integrity(case_id, raw_bytes=bytes(tampered))
        assert bad_check["integrity_verified"] is False
        assert bad_check["status"] == "MISMATCH_TAMPERED"


def test_non_destructive_legacy_schema_migration():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # 1. Create pre-Phase 8 16-column legacy SQLite DB with existing records
        conn = sqlite3.connect(tmp.name)
        conn.execute(
            """
            CREATE TABLE forensic_examinations (
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
                model_version TEXT NOT NULL
            )
            """
        )
        legacy_rows = [
            (
                "DFS-1700000000001",
                "2026-11-01 10:00:00 UTC",
                "legacy_portrait_1.jpg",
                "a" * 64,
                "High Authenticity",
                "AUTHENTIC REAL HUMAN",
                0.12,
                88.0,
                0.12,
                0.15,
                0.18,
                82.4,
                "Mid-Face",
                0,
                "MPS",
                "DualDomainSentinel-v1.0",
            ),
            (
                "DFS-1700000000002",
                "2026-11-02 12:30:00 UTC",
                "legacy_deepfake_2.jpg",
                "b" * 64,
                "Likely Manipulated",
                "DEEPFAKE / AI-GENERATED",
                0.91,
                91.0,
                0.91,
                0.84,
                0.79,
                94.1,
                "Eye / Brow",
                1,
                "CPU",
                "DualDomainSentinel-v1.0",
            ),
        ]
        conn.executemany(
            "INSERT INTO forensic_examinations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            legacy_rows,
        )
        conn.commit()
        conn.close()

        # 2. Initialize upgraded ForensicDatabase on legacy file (must preserve rows)
        db = ForensicDatabase(db_path=tmp.name)
        info = db.get_schema_info()
        for required_col in (
            "evidence_id",
            "media_type",
            "raw_file_sha256",
            "file_size_bytes",
            "media_dimensions",
            "decision_threshold",
            "calibrated_temperature",
            "prediction_outcome",
            "processing_status",
            "is_archived",
        ):
            assert required_col in info["forensic_examinations_columns"]
        assert "audit_events" in info["audit_events_columns"] or len(info["audit_events_columns"]) >= 7
        assert info["total_examinations"] == 2

        history = db.get_history(limit=10)
        assert len(history) == 2
        by_id = {r["case_id"]: r for r in history}
        assert by_id["DFS-1700000000001"]["image_name"] == "legacy_portrait_1.jpg"
        assert by_id["DFS-1700000000001"]["image_sha256"] == "a" * 64
        assert by_id["DFS-1700000000001"]["evidence_id"].startswith("EVD-LEGACY-")
        assert by_id["DFS-1700000000002"]["fake_probability"] == pytest.approx(0.91)
        assert by_id["DFS-1700000000002"]["is_archived"] == 0

        # Idempotent re-init must never duplicate or alter preserved records
        db_reinit = ForensicDatabase(db_path=tmp.name)
        assert len(db_reinit.get_history(limit=10)) == 2


def test_immutable_audit_events_and_confirmation_guarded_archiving():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = ForensicDatabase(db_path=tmp.name)
        raw_bytes = _make_jpeg_bytes(color=(10, 40, 200))

        case_id = db.log_examination(
            {
                "aligned_face": Image.new("RGB", (256, 256), color=(10, 40, 200)),
                "verdict_tier": "AUTHENTIC",
                "prediction_outcome": "AUTHENTIC",
                "prediction_label": "AUTHENTIC",
                "fake_probability": 0.22,
                "confidence_percent": 78.0,
                "inference_time_ms": 48.0,
                "calibration_metadata": {"decision_threshold": 0.71, "temperature": 2.6543},
            },
            image_source_name="clean_portrait.jpg",
            device_str="MPS",
            raw_file_bytes=raw_bytes,
            media_dimensions="320x240",
        )
        rec = db.get_record(case_id)
        evd_id = rec["evidence_id"]

        db.record_report_generation(
            case_id=case_id,
            evidence_id=evd_id,
            report_format="pdf",
            report_bytes=b"%PDF-1.4 mock bytes",
            output_name="mock_report.pdf",
        )

        events = db.get_audit_events(evidence_id=evd_id)
        event_types = [e["event_type"] for e in events]
        assert event_types == [
            EVENT_UPLOAD,
            EVENT_PROCESSING,
            EVENT_PREDICTION,
            EVENT_REPORT_GENERATION,
        ]

        # SQLite immutability triggers block UPDATE and DELETE on audit_events
        conn = sqlite3.connect(tmp.name)
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("UPDATE audit_events SET event_type = 'tampered'")
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("DELETE FROM audit_events")
        conn.close()

        # Unknown audit event type is rejected
        with pytest.raises(ValueError):
            db.log_audit_event(evidence_id=evd_id, event_type="arbitrary_type")

        # Unconfirmed clear_history MUST fail (B9 fix)
        with pytest.raises(ValueError, match="confirm=True"):
            db.clear_history()
        assert len(db.get_history()) == 1

        # Confirmed archive removes from active view while keeping non-destructive record + audit event
        archived_count = db.clear_history(confirm=True, archive_only=True, actor="test_auditor")
        assert archived_count == 1
        assert len(db.get_history()) == 0
        assert len(db.get_history(include_archived=True)) == 1

        archive_events = db.get_audit_events(event_type=EVENT_ARCHIVE_ACTION)
        assert len(archive_events) == 1
        assert archive_events[0]["details"]["affected_records"] == 1


def test_video_examination_logging_and_custody_disclaimer():
    assert "software forensic audit" in CHAIN_OF_CUSTODY_DISCLAIMER.lower()
    assert "legally admissible chain of custody" in CHAIN_OF_CUSTODY_DISCLAIMER.lower()

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = ForensicDatabase(db_path=tmp.name)
        vid_sha = "c" * 64
        mock_video_res = {
            "status": "completed",
            "filename": "clip_sample.mp4",
            "sha256": vid_sha,
            "file_metadata": {
                "filename": "clip_sample.mp4",
                "file_size_bytes": 124800,
                "width": 640,
                "height": 480,
                "fps": 25.0,
                "frame_count": 50,
                "duration_sec": 2.0,
            },
            "fps": 25.0,
            "frame_count": 50,
            "duration_sec": 2.0,
            "processed_frames": 4,
            "detected_faces": 4,
            "frames_with_faces": 4,
            "authentic_frame_count": 1,
            "manipulated_frame_count": 3,
            "uncertain_frame_count": 0,
            "overall_assessment": "MANIPULATED",
            "prediction_outcome": "MANIPULATED",
            "aggregated_fake_probability": 0.82,
            "aggregated_confidence_percent": 82.0,
            "suspicious_timestamps": [{"start_timestamp": "00:00.00", "end_timestamp": "00:01.50"}],
            "analysis_scope": "frame_level_inference",
            "temporal_modeling": False,
            "calibration_metadata": {"temperature": 2.6543, "decision_threshold": 0.71},
            "inference_time_ms": 210.5,
            "timeline": [],
        }

        case_id = db.log_video_examination(mock_video_res, device_str="CPU")
        rec = db.get_record(case_id)
        assert rec["media_type"] == "video"
        assert rec["raw_file_sha256"] == vid_sha
        assert rec["file_size_bytes"] == 124800
        assert "640x480 @ 25.00fps" in rec["media_dimensions"]
        assert rec["decision_threshold"] == pytest.approx(0.71)
        assert rec["prediction_outcome"] == "MANIPULATED"
        assert db.get_summary_stats()["videos"] == 1
