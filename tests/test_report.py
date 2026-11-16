"""
Unit and multi-page integration tests for the upgraded Phase 9 PDF Forensic Certificate Generator.
"""

import os
import tempfile
import numpy as np
from PIL import Image
from reportlab.lib import colors

from app.report_generator import (
    _resolve_verdict_theme,
    compute_sha256,
    generate_pdf_report,
    generate_video_pdf_report,
)


def test_b8_verdict_color_resolution_and_uncertain_safety():
    amber = colors.HexColor("#d97706")
    red = colors.HexColor("#b91c1c")
    green = colors.HexColor("#047857")

    # B8 regression check: UNCERTAIN / Inconclusive must NEVER turn red even if score >= 0.50 or is_fake flag stale
    c_unc, lbl_unc = _resolve_verdict_theme(
        {
            "prediction_outcome": "UNCERTAIN / NEEDS MANUAL REVIEW",
            "verdict_tier": "UNCERTAIN / NEEDS MANUAL REVIEW",
            "is_uncertain": True,
            "is_fake": True,
        }
    )
    assert c_unc == amber
    assert "UNCERTAIN" in lbl_unc

    c_inc, _ = _resolve_verdict_theme({"verdict_tier": "Inconclusive", "is_fake": False})
    assert c_inc == amber

    c_fake, _ = _resolve_verdict_theme(
        {"prediction_outcome": "MANIPULATED", "verdict_tier": "MANIPULATED", "is_fake": True, "is_uncertain": False}
    )
    assert c_fake == red

    c_real, _ = _resolve_verdict_theme(
        {"prediction_outcome": "AUTHENTIC", "verdict_tier": "AUTHENTIC", "is_fake": False, "is_uncertain": False}
    )
    assert c_real == green


def test_image_pdf_report_generation_multiface_and_events():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name

    try:
        dummy_img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        h = compute_sha256(dummy_img)
        assert len(h) == 64

        freqs = np.arange(0, 64, dtype=float)
        profile = np.maximum(0.1, 8.0 - 1.5 * np.log1p(freqs))

        mock_results = {
            "case_id": "DFS-TEST-12345",
            "evidence_id": "EVD-TEST-99999",
            "raw_file_sha256": "e" * 64,
            "file_size_bytes": 48200,
            "media_dimensions": "1024x768",
            "aligned_face": dummy_img,
            "gradcam_overlay": dummy_img,
            "fft_spectrum_img": dummy_img,
            "fft_log_colormap_img": dummy_img,
            "ela_image": dummy_img,
            "ela_heatmap_color": dummy_img,
            "ela_mean_error": 7.45,
            "ela_details": {"quad_discrepancy_ratio": 0.215},
            "azimuthal_frequencies": freqs,
            "azimuthal_profile": profile,
            "spectral_details": {"detected_peaks": [12, 24], "has_deconv_spikes": True},
            "spectral_comparison": {"ideal_decay": profile},
            "is_fake": False,
            "is_uncertain": True,
            "prediction_outcome": "UNCERTAIN / NEEDS MANUAL REVIEW",
            "fake_probability": 0.712,
            "confidence_percent": 71.2,
            "verdict_tier": "UNCERTAIN / NEEDS MANUAL REVIEW",
            "verdict_badge": "⚖️ VERDICT: UNCERTAIN / NEEDS MANUAL REVIEW",
            "verdict_explanation": "Prediction lies within the validated ambiguity band.",
            "inference_time_ms": 105.4,
            "face_detected": True,
            "faces_count": 2,
            "calibration_metadata": {
                "temperature": 2.6543,
                "decision_threshold": 0.71,
                "uncertain_low": 0.707,
                "uncertain_high": 0.7188,
            },
            "forensic_evidence": {
                "neural_fake_prob": 0.712,
                "neural_uncalibrated_prob": 0.91,
                "spectral_anomaly_prob": 0.65,
                "ela_anomaly_prob": 0.52,
            },
            "multi_face_reports": [
                {
                    "face_index": 0,
                    "is_primary": True,
                    "bbox": (10, 10, 200, 200),
                    "prediction_outcome": "UNCERTAIN / NEEDS MANUAL REVIEW",
                    "fake_probability": 0.712,
                    "confidence_percent": 71.2,
                    "spatial_landmarks": {"dominant_region": "Upper / Eye Region"},
                },
                {
                    "face_index": 1,
                    "is_primary": False,
                    "bbox": (300, 40, 180, 180),
                    "prediction_outcome": "AUTHENTIC",
                    "fake_probability": 0.34,
                    "confidence_percent": 66.0,
                    "spatial_landmarks": {"dominant_region": "Mid-Face"},
                },
            ],
            "forensic_observations": [
                "Primary Classifier routed to manual review band.",
                "Grad-CAM highlights are advisory only.",
            ],
        }
        mock_events = [
            {
                "event_id": "EVT-1",
                "event_type": "upload",
                "event_timestamp_utc": "2026-11-12 05:00:00.000 UTC",
                "sha256_at_event": "e" * 64,
                "actor": "pytest",
            },
            {
                "event_id": "EVT-2",
                "event_type": "prediction",
                "event_timestamp_utc": "2026-11-12 05:00:01.000 UTC",
                "sha256_at_event": "e" * 64,
                "actor": "pytest",
            },
        ]

        out = generate_pdf_report(mock_results, output_path=pdf_path, audit_events=mock_events)
        assert out == pdf_path
        assert os.path.exists(pdf_path)
        pdf_bytes = open(pdf_path, "rb").read()
        assert len(pdf_bytes) > 5000
        assert b"%PDF" in pdf_bytes[:16]
        assert b"0.50N + 0.30S + 0.20E" not in pdf_bytes
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_video_multipage_pdf_report_generation():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name

    try:
        # 50 sampled timeline rows triggers multi-page pagination cleanly
        timeline = [
            {
                "frame_index": i * 5,
                "timestamp_sec": i * 0.2,
                "timestamp_label": f"00:{i * 0.2:05.2f}",
                "has_face": True,
                "detected_face_count": 1,
                "fake_probability": 0.85 if 10 <= i <= 18 else 0.25,
                "prediction_outcome": "MANIPULATED" if 10 <= i <= 18 else "AUTHENTIC",
                "timeline_color": "red" if 10 <= i <= 18 else "green",
            }
            for i in range(50)
        ]
        mock_video_res = {
            "case_id": "DFS-VID-001",
            "evidence_id": "EVD-VID-001",
            "filename": "suspect_clip.mp4",
            "sha256": "f" * 64,
            "status": "completed",
            "fps": 25.0,
            "frame_count": 250,
            "duration_sec": 10.0,
            "duration_formatted": "00:10.00",
            "processed_frames": 50,
            "detected_faces": 50,
            "frames_with_faces": 50,
            "frames_without_faces": 0,
            "authentic_frame_count": 41,
            "manipulated_frame_count": 9,
            "uncertain_frame_count": 0,
            "authentic_frame_pct": 82.0,
            "manipulated_frame_pct": 18.0,
            "uncertain_frame_pct": 0.0,
            "overall_assessment": "UNCERTAIN / NEEDS MANUAL REVIEW",
            "prediction_outcome": "UNCERTAIN / NEEDS MANUAL REVIEW",
            "is_fake": False,
            "is_uncertain": True,
            "verdict_badge": "⚖️ VIDEO ASSESSMENT: UNCERTAIN / NEEDS MANUAL REVIEW",
            "verdict_explanation": "Localized suspicious segment from 00:02.00 to 00:03.60.",
            "aggregated_fake_probability": 0.36,
            "aggregated_confidence_percent": 64.0,
            "median_fake_probability": 0.25,
            "p90_fake_probability": 0.85,
            "sustained_quorum_required": 15,
            "isolated_spike_suppressed": False,
            "inference_time_ms": 420.0,
            "analysis_scope": "frame_level_inference",
            "temporal_modeling": False,
            "file_metadata": {"width": 1280, "height": 720, "file_size_mb": 3.4},
            "sampling_config": {"sample_mode": "fps"},
            "calibration_metadata": {
                "temperature": 2.6543,
                "decision_threshold": 0.71,
                "uncertain_low": 0.707,
                "uncertain_high": 0.7188,
            },
            "suspicious_timestamps": [
                {
                    "start_timestamp": "00:02.00",
                    "end_timestamp": "00:03.60",
                    "duration_sec": 1.6,
                    "frame_count": 9,
                    "severity": "High Confidence",
                    "peak_fake_probability": 0.85,
                }
            ],
            "timeline": timeline,
        }

        out = generate_video_pdf_report(mock_video_res, output_path=pdf_path)
        assert out == pdf_path
        assert os.path.exists(pdf_path)
        data = open(pdf_path, "rb").read()
        assert len(data) > 5000
        # Multi-page check (>1 /Type /Page occurrence)
        assert data.count(b"/Type /Page") >= 2
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
