"""
Unit & Integration Tests for Phase 6: Frame-Level Video Deepfake Detection,
Validation, Multi-Frame Aggregation Safeguards, Timeline & Report Generation.
"""

import json
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from app.video_dashboard import plot_confidence_distribution, plot_video_timeline
from src.evaluation.calibration import (
    OUTCOME_AUTHENTIC,
    OUTCOME_MANIPULATED,
    OUTCOME_UNCERTAIN,
)
from src.pipeline import ForensicInferencePipeline
from src.video.video_analyzer import (
    FRAME_OUTCOME_NO_FACE,
    VideoForensicAnalyzer,
    aggregate_video_scores,
    build_video_forensic_report,
    extract_suspicious_intervals,
    format_video_forensic_markdown,
    validate_video_file,
)


def _write_portrait_video(
    out_path: Path,
    fourcc_str: str,
    image_paths: List[Path],
    fps: float = 10.0,
    repeats_per_image: int = 5,
    frame_size=(384, 384),
) -> Path:
    """Encodes a real playable video containing embedded portrait faces in MP4/AVI/MOV."""
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    assert writer.isOpened(), f"OpenCV failed to open VideoWriter for {out_path} ({fourcc_str})"
    try:
        for img_path in image_paths:
            bgr = cv2.imread(str(img_path))
            assert bgr is not None, f"Failed to read {img_path}"
            canvas = np.full((frame_size[1], frame_size[0], 3), 210, dtype=np.uint8)
            pad = 64
            face_resized = cv2.resize(bgr, (frame_size[0] - 2 * pad, frame_size[1] - 2 * pad))
            canvas[pad:frame_size[1] - pad, pad:frame_size[0] - pad] = face_resized
            for _ in range(repeats_per_image):
                writer.write(canvas)
    finally:
        writer.release()
    return out_path


def _write_blank_video(out_path: Path, num_frames: int = 12, fps: float = 12.0, frame_size=(320, 240)) -> Path:
    """Encodes an MP4 containing zero faces (plain gradient frames)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    assert writer.isOpened()
    try:
        for i in range(num_frames):
            frame = np.full((frame_size[1], frame_size[0], 3), (i * 15) % 240, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return out_path


def test_validate_video_file_formats_and_corruption(tmp_path):
    demo = sorted(Path("data/demo_samples").glob("*.jpg"))[:2]
    mp4_path = _write_portrait_video(tmp_path / "sample.mp4", "mp4v", demo, fps=10.0, repeats_per_image=4)
    avi_path = _write_portrait_video(tmp_path / "sample.avi", "MJPG", demo, fps=10.0, repeats_per_image=4)
    mov_path = _write_portrait_video(tmp_path / "sample.mov", "mp4v", demo, fps=10.0, repeats_per_image=4)

    for vp in (mp4_path, avi_path, mov_path):
        ok, err, meta = validate_video_file(vp)
        assert ok is True, f"Expected {vp.name} valid, got {err}"
        assert err is None
        assert meta["sha256"] and len(meta["sha256"]) == 64
        assert meta["frame_count"] >= 6
        assert meta["duration_sec"] > 0.0

    # Missing file
    ok_m, err_m, _ = validate_video_file(tmp_path / "missing.mp4")
    assert ok_m is False and "not found" in err_m.lower()

    # Unsupported extension
    txt = tmp_path / "notes.txt"
    txt.write_text("not a video")
    ok_ext, err_ext, _ = validate_video_file(txt)
    assert ok_ext is False and "unsupported video format" in err_ext.lower()

    # 0-byte file
    empty_mp4 = tmp_path / "empty.mp4"
    empty_mp4.write_bytes(b"")
    ok_empty, err_empty, _ = validate_video_file(empty_mp4)
    assert ok_empty is False and "empty" in err_empty.lower()

    # Corrupted garbage bytes masquerading as MP4
    bad_mp4 = tmp_path / "corrupt.mp4"
    bad_mp4.write_bytes(b"NOT_A_REAL_VIDEO_STREAM_" * 100)
    ok_corrupt, err_corrupt, _ = validate_video_file(bad_mp4)
    assert ok_corrupt is False and err_corrupt is not None

    # Exceeded max file size check
    ok_size, err_size, _ = validate_video_file(mp4_path, max_size_mb=0.0001)
    assert ok_size is False and "exceeds maximum limit" in err_size.lower()


def test_single_suspicious_frame_never_marks_video_manipulated():
    policy = {
        "decision_threshold": 0.71,
        "uncertain_low": 0.707,
        "uncertain_high": 0.7188,
    }

    # Case A: 9 clearly authentic frames + 1 high-probability spike -> must NOT be MANIPULATED
    records_10 = [
        {
            "has_face": True,
            "fake_probability": 0.35,
            "classifier_confidence": 0.65,
            "prediction_outcome": OUTCOME_AUTHENTIC,
        }
        for _ in range(9)
    ] + [
        {
            "has_face": True,
            "fake_probability": 0.722,
            "classifier_confidence": 0.722,
            "prediction_outcome": OUTCOME_MANIPULATED,
        }
    ]
    agg_10 = aggregate_video_scores(records_10, policy)
    assert agg_10["prediction_outcome"] == OUTCOME_AUTHENTIC
    assert agg_10["is_fake"] is False
    assert agg_10["isolated_spike_suppressed"] is True

    # Case B: Small clip (2 authentic + 1 manipulated) -> must route to UNCERTAIN, NEVER MANIPULATED
    records_3 = records_10[7:]
    agg_3 = aggregate_video_scores(records_3, policy)
    assert agg_3["prediction_outcome"] == OUTCOME_UNCERTAIN
    assert agg_3["is_fake"] is False
    assert agg_3["isolated_spike_suppressed"] is True

    # Case C: Only 1 evaluated frame even at high prob -> UNCERTAIN
    agg_1 = aggregate_video_scores([records_10[-1]], policy)
    assert agg_1["prediction_outcome"] == OUTCOME_UNCERTAIN
    assert agg_1["is_fake"] is False

    # Case D: Zero face frames -> NO_FACES_DETECTED
    agg_0 = aggregate_video_scores(
        [{"has_face": False, "fake_probability": None, "prediction_outcome": FRAME_OUTCOME_NO_FACE}],
        policy,
    )
    assert agg_0["status"] == "no_faces_detected"
    assert agg_0["prediction_outcome"] == FRAME_OUTCOME_NO_FACE
    assert agg_0["is_fake"] is False

    # Case E: Sustained multi-frame manipulation (7/10 above policy band) -> MANIPULATED
    records_fake = [
        {
            "has_face": True,
            "fake_probability": 0.721,
            "classifier_confidence": 0.721,
            "prediction_outcome": OUTCOME_MANIPULATED,
        }
        for _ in range(7)
    ] + [
        {
            "has_face": True,
            "fake_probability": 0.35,
            "classifier_confidence": 0.65,
            "prediction_outcome": OUTCOME_AUTHENTIC,
        }
        for _ in range(3)
    ]
    agg_fake = aggregate_video_scores(records_fake, policy)
    assert agg_fake["prediction_outcome"] == OUTCOME_MANIPULATED
    assert agg_fake["is_fake"] is True


def test_extract_suspicious_intervals_groups_timestamps():
    recs = [
        {"frame_index": 0, "timestamp_sec": 0.0, "has_face": True, "prediction_outcome": OUTCOME_AUTHENTIC, "fake_probability": 0.32},
        {"frame_index": 10, "timestamp_sec": 1.0, "has_face": True, "prediction_outcome": OUTCOME_MANIPULATED, "fake_probability": 0.72},
        {"frame_index": 20, "timestamp_sec": 2.0, "has_face": True, "prediction_outcome": OUTCOME_MANIPULATED, "fake_probability": 0.73},
        {"frame_index": 30, "timestamp_sec": 3.0, "has_face": True, "prediction_outcome": OUTCOME_AUTHENTIC, "fake_probability": 0.31},
        {"frame_index": 40, "timestamp_sec": 4.0, "has_face": True, "prediction_outcome": OUTCOME_UNCERTAIN, "fake_probability": 0.712},
    ]
    intervals = extract_suspicious_intervals(recs, sample_interval_sec=1.0)
    assert len(intervals) == 2
    assert intervals[0]["severity"] == OUTCOME_MANIPULATED
    assert intervals[0]["frame_count"] == 2
    assert intervals[0]["start_sec"] == 1.0 and intervals[0]["end_sec"] == 2.0
    assert intervals[1]["severity"] == OUTCOME_UNCERTAIN
    assert intervals[1]["frame_count"] == 1


@pytest.mark.parametrize(
    "ext,fourcc,mode",
    [
        (".mp4", "mp4v", "fps"),
        (".avi", "MJPG", "interval"),
        (".mov", "mp4v", "uniform"),
    ],
)
def test_end_to_end_video_analyzer_mp4_avi_mov(tmp_path, ext, fourcc, mode):
    samples = [
        Path("data/demo_samples/authentic_studio_photo_1.jpg"),
        Path("data/demo_samples/deepfake_stylegan_checkerboard.jpg"),
    ]
    vid_path = _write_portrait_video(
        tmp_path / f"portrait_test{ext}",
        fourcc_str=fourcc,
        image_paths=samples,
        fps=8.0,
        repeats_per_image=4,
    )

    ckpt = "checkpoints/best_model.pth" if Path("checkpoints/best_model.pth").exists() else None
    pipeline = ForensicInferencePipeline(checkpoint_path=ckpt, device="cpu")
    analyzer = VideoForensicAnalyzer(pipeline=pipeline)

    res = analyzer.analyze_video(
        vid_path,
        sample_mode=mode,
        sample_fps=2.0,
        sample_every_n_frames=2,
        max_frames=6,
    )

    assert res["status"] == "completed"
    assert res["filename"] == f"portrait_test{ext}"
    assert res["processed_frames"] >= 2
    assert res["detected_faces"] >= 2
    assert res["duration_sec"] > 0.0
    assert res["frame_count"] >= 6
    assert abs(res["authentic_frame_pct"] + res["manipulated_frame_pct"] + res["uncertain_frame_pct"] - 100.0) < 0.2
    assert res["prediction_outcome"] in {OUTCOME_AUTHENTIC, OUTCOME_MANIPULATED, OUTCOME_UNCERTAIN}
    assert res["analysis_scope"] == "frame_level_inference"
    assert res["temporal_modeling"] is False
    assert "trimmed mean" in res["aggregation_method"].lower()
    assert len(res["timeline"]) == res["processed_frames"]
    assert all(t["timeline_color"] in {"green", "red", "yellow", "gray"} for t in res["timeline"])
    assert any(t["frame_preview"] is not None for t in res["timeline"])

    # Structured report must be 100% JSON serializable and Markdown-renderable
    report = build_video_forensic_report(res)
    serialized = json.dumps(report)
    assert len(serialized) > 100
    md = format_video_forensic_markdown(report)
    assert "Methodological Scope" in md and "temporal_modeling=False" in md

    # Interactive Matplotlib dashboard charts must render cleanly without exceptions
    fig_tl = plot_video_timeline(res["timeline"], res["calibration_metadata"], selected_pos=0)
    fig_dist = plot_confidence_distribution(res)
    plt.close(fig_tl)
    plt.close(fig_dist)


def test_video_analyzer_handles_no_face_and_corrupted_videos(tmp_path):
    pipeline = ForensicInferencePipeline(checkpoint_path=None, device="cpu")
    analyzer = VideoForensicAnalyzer(pipeline=pipeline)

    # 1. No-face clip
    blank_mp4 = _write_blank_video(tmp_path / "no_faces.mp4", num_frames=10, fps=10.0)
    res_nf = analyzer.analyze_video(blank_mp4, sample_mode="uniform", max_frames=4, allow_center_crop_fallback=False)
    assert res_nf["status"] == "no_faces_detected"
    assert res_nf["detected_faces"] == 0
    assert res_nf["prediction_outcome"] == FRAME_OUTCOME_NO_FACE
    assert all(item["timeline_color"] == "gray" for item in res_nf["timeline"])

    # 2. Corrupted clip
    corrupt_mp4 = tmp_path / "broken.mp4"
    corrupt_mp4.write_bytes(b"corrupted_binary_stream")
    res_err = analyzer.analyze_video(corrupt_mp4)
    assert res_err["status"] == "error"
    assert res_err["prediction_outcome"] == "ERROR"
    assert res_err["error"] is not None
