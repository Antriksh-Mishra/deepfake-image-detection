"""
Phase 11 Full End-to-End QA Test Suite.
Validates remaining E2E integration journeys across calibrated image inference,
threshold overrides, image mode/size edge cases, multi-face composite -> DB -> PDF,
video pipeline -> DB -> PDF, missing checkpoints, corrupted files, Streamlit AppTest
interactive multi-section workflows (in isolated DB/report paths), and baseline integrity.
"""

import hashlib
import io
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.report_generator import generate_pdf_report, generate_video_pdf_report
from src.core.database import (
    EVENT_PREDICTION,
    EVENT_PROCESSING,
    EVENT_REPORT_GENERATION,
    EVENT_UPLOAD,
    ForensicDatabase,
    compute_bytes_sha256,
)
from src.data.dataset import DualDomainDataset
from src.evaluation.calibration import (
    OUTCOME_AUTHENTIC,
    OUTCOME_MANIPULATED,
    OUTCOME_UNCERTAIN,
)
from src.evaluation.external_benchmark import _load_model_for_external_eval
from src.pipeline import ForensicInferencePipeline
from src.video.video_analyzer import VideoForensicAnalyzer


CKPT_PATH = Path("checkpoints/best_model.pth")
BASELINE_CKPT_SHA256 = "e8c3f62ac628323f0a3daee52923eee611da7850bf7c05c5481dc44ff49cde00"


def _active_ckpt_arg() -> str | None:
    return str(CKPT_PATH) if CKPT_PATH.exists() else None


def test_calibrated_image_authentic_vs_manipulated_and_threshold_routing():
    pipe = ForensicInferencePipeline(checkpoint_path=_active_ckpt_arg(), device="cpu")
    auth_p = Path("data/demo_samples/authentic_studio_photo_1.jpg")
    diff_p = Path("data/demo_samples/deepfake_diffusion_ai_face.jpg")
    gan_p = Path("data/demo_samples/deepfake_stylegan_checkerboard.jpg")
    if not (CKPT_PATH.exists() and auth_p.exists() and diff_p.exists() and gan_p.exists()):
        pytest.skip("Production checkpoint or demo samples unavailable")

    with Image.open(auth_p) as im_a:
        res_auth = pipe.analyze_image(im_a.convert("RGB"))
    with Image.open(diff_p) as im_d:
        res_diff = pipe.analyze_image(im_d.convert("RGB"))
    with Image.open(gan_p) as im_g:
        res_gan = pipe.analyze_image(im_g.convert("RGB"))

    assert res_auth["prediction_outcome"] == OUTCOME_AUTHENTIC
    assert res_diff["prediction_outcome"] == OUTCOME_MANIPULATED
    assert res_diff["fake_probability"] > res_auth["fake_probability"]
    assert res_gan["fake_probability"] > res_auth["fake_probability"]
    assert res_gan["prediction_outcome"] in {OUTCOME_UNCERTAIN, OUTCOME_MANIPULATED}

    # Explicit threshold override shifts outcome boundaries deterministically
    with Image.open(auth_p) as im_a:
        rgb_auth = im_a.convert("RGB")
        forced_manip = pipe.analyze_image(rgb_auth, threshold=0.20)
        forced_auth = pipe.analyze_image(rgb_auth, threshold=0.90)
    assert forced_manip["prediction_outcome"] == OUTCOME_MANIPULATED
    assert forced_auth["prediction_outcome"] == OUTCOME_AUTHENTIC


@pytest.mark.parametrize(
    "mode,size",
    [
        ("L", (256, 256)),
        ("RGBA", (300, 200)),
        ("CMYK", (256, 256)),
        ("RGB", (1, 1)),
        ("RGB", (60, 60)),
    ],
)
def test_edge_color_modes_tiny_resolutions_and_quality_guards(mode, size):
    pipe = ForensicInferencePipeline(checkpoint_path=_active_ckpt_arg(), device="cpu")
    color = 128 if mode == "L" else (128, 128, 128, 255) if mode in {"RGBA", "CMYK"} else (128, 128, 128)
    edge_img = Image.new(mode, size, color=color)

    res = pipe.analyze_image(edge_img)
    assert res["prediction_outcome"] == OUTCOME_UNCERTAIN
    assert res["is_uncertain"] is True
    assert res["uncertainty_reason"] is not None
    assert "sharpness" in res["uncertainty_reason"].lower() or "resolution" in res["uncertainty_reason"].lower()


def test_multiface_three_faces_pipeline_db_and_real_pdf_e2e(tmp_path: Path):
    samples = [
        Path("data/demo_samples/authentic_studio_photo_1.jpg"),
        Path("data/demo_samples/deepfake_stylegan_checkerboard.jpg"),
        Path("data/demo_samples/deepfake_diffusion_ai_face.jpg"),
    ]
    if not all(p.exists() for p in samples):
        pytest.skip("Demo portraits required for 3-face composite check")

    canvas = Image.new("RGB", (980, 360), color=(35, 55, 80))
    for i, sp in enumerate(samples):
        with Image.open(sp) as im:
            canvas.paste(im.convert("RGB").resize((240, 240)), (40 + i * 330, 60))

    raw_buf = io.BytesIO()
    canvas.save(raw_buf, format="JPEG", quality=93)
    raw_bytes = raw_buf.getvalue()
    raw_sha = compute_bytes_sha256(raw_bytes)

    pipe = ForensicInferencePipeline(checkpoint_path=_active_ckpt_arg(), device="cpu")
    res = pipe.analyze_image(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

    assert res["faces_count"] >= 2
    assert len(res["multi_face_reports"]) == res["faces_count"]
    assert res["performance_telemetry"]["batched_inference"] is True
    assert res["performance_telemetry"]["batch_size"] == res["faces_count"]
    for rep in res["multi_face_reports"]:
        assert rep["gradcam_overlay"].size == (256, 256)
        assert rep["fft_log_colormap_img"].size == (256, 256)
        assert rep["ela_heatmap_color"].size == (256, 256)

    db = ForensicDatabase(db_path=str(tmp_path / "qa_img.db"))
    intake = db.record_media_upload(
        filename="three_faces_scene.jpg",
        raw_file_bytes=raw_bytes,
        media_type="image",
        media_dimensions=f"{canvas.width}x{canvas.height}",
        actor="pytest_qa",
    )
    case_id = db.log_examination(
        res,
        image_source_name="three_faces_scene.jpg",
        device_str="CPU",
        raw_file_bytes=raw_bytes,
        file_size_bytes=len(raw_bytes),
        media_type="image",
        media_dimensions=f"{canvas.width}x{canvas.height}",
        evidence_id=intake["evidence_id"],
        upload_already_logged=True,
        actor="pytest_qa",
    )
    res["case_id"] = case_id
    res["evidence_id"] = intake["evidence_id"]
    res["raw_file_sha256"] = raw_sha

    # Verify DB raw hash vs crop hash + tamper detection
    assert db.verify_evidence_integrity(case_id, raw_bytes=raw_bytes)["integrity_verified"] is True
    assert db.verify_evidence_integrity(case_id, raw_bytes=raw_bytes[:-1] + b"X")["status"] == "MISMATCH_TAMPERED"

    # Compile real ReportLab PDF from genuine pipeline output
    events = db.get_audit_events(evidence_id=intake["evidence_id"])
    assert [e["event_type"] for e in events] == [EVENT_UPLOAD, EVENT_PROCESSING, EVENT_PREDICTION]

    pdf_out = tmp_path / "real_img_certificate.pdf"
    generate_pdf_report(res, output_path=str(pdf_out), audit_events=events)
    pdf_bytes = pdf_out.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000

    db.record_report_generation(
        case_id=case_id,
        evidence_id=intake["evidence_id"],
        report_format="pdf",
        report_bytes=pdf_bytes,
        output_name=pdf_out.name,
        actor="pytest_qa",
    )
    assert len(db.get_audit_events(evidence_id=intake["evidence_id"], event_type=EVENT_REPORT_GENERATION)) == 1


def test_real_video_pipeline_to_db_to_multipage_pdf_e2e(tmp_path: Path):
    samples = [
        Path("data/demo_samples/authentic_studio_photo_1.jpg"),
        Path("data/demo_samples/deepfake_diffusion_ai_face.jpg"),
    ]
    if not all(p.exists() for p in samples):
        pytest.skip("Demo portraits required for synthetic portrait video test")

    vid_path = tmp_path / "qa_clip.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (384, 384))
    assert writer.isOpened()
    try:
        for sp in samples:
            bgr = cv2.resize(cv2.imread(str(sp)), (256, 256))
            canvas = np.full((384, 384, 3), 205, dtype=np.uint8)
            canvas[64:320, 64:320] = bgr
            for _ in range(4):
                writer.write(canvas)
    finally:
        writer.release()

    analyzer = VideoForensicAnalyzer(checkpoint_path=_active_ckpt_arg(), device="cpu")
    res = analyzer.analyze_video(vid_path, sample_mode="uniform", max_frames=4, batch_size=2)
    assert res["status"] == "completed"
    assert res["temporal_modeling"] is False
    assert res["performance_telemetry"]["batched_inference"] is True

    db = ForensicDatabase(db_path=str(tmp_path / "qa_vid.db"))
    case_id = db.log_video_examination(res, device_str="CPU", actor="pytest_qa")
    res["case_id"] = case_id
    rec = db.get_record(case_id)
    assert rec is not None and rec["media_type"] == "video"

    events = db.get_audit_events(evidence_id=rec["evidence_id"])
    pdf_out = tmp_path / "real_vid_certificate.pdf"
    generate_video_pdf_report(res, output_path=str(pdf_out), audit_events=events)
    pdf_bytes = pdf_out.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000


def test_corrupted_missing_files_and_missing_checkpoint_fallback(tmp_path: Path):
    # 1. Pipeline & VideoAnalyzer gracefully fall back to initialized weights on missing checkpoint path
    missing_ckpt = str(tmp_path / "missing_checkpoint.pth")
    pipe = ForensicInferencePipeline(checkpoint_path=missing_ckpt, device="cpu")
    res = pipe.analyze_image(Image.new("RGB", (256, 256), color=(140, 110, 90)))
    assert 0.0 <= res["fake_probability"] <= 1.0

    vid = VideoForensicAnalyzer(checkpoint_path=missing_ckpt, device="cpu")
    assert vid.device.type == "cpu"

    # 2. External benchmark strictly rejects missing checkpoints (never benchmarks unverified weights)
    with pytest.raises(FileNotFoundError):
        _load_model_for_external_eval(Path(missing_ckpt), {}, torch_device := __import__("torch").device("cpu"))
    del torch_device

    # 3. Missing / corrupted image paths handled cleanly in DualDomainDataset
    missing_img = tmp_path / "does_not_exist.jpg"
    corrupt_img = tmp_path / "corrupt.jpg"
    corrupt_img.write_bytes(b"corrupt_stream")
    ds = DualDomainDataset([str(missing_img), str(corrupt_img)], [0, 1], is_train=False, img_size=128)
    s0 = ds[0]
    s1 = ds[1]
    assert s0["is_corrupt"] is True and s1["is_corrupt"] is True
    assert len(ds.corrupted_paths) == 2

    # 4. Missing DB case raises KeyError cleanly
    db = ForensicDatabase(db_path=str(tmp_path / "empty.db"))
    with pytest.raises(KeyError):
        db.verify_evidence_integrity("DFS-DOES-NOT-EXIST", raw_bytes=b"123")


def test_streamlit_e2e_six_sections_and_interactive_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    isolated_db = tmp_path / "ui_e2e.db"
    isolated_pdf = tmp_path / "ui_forensic_report.pdf"
    monkeypatch.setenv("DFAKE_DB_PATH", str(isolated_db))
    monkeypatch.setenv("DFAKE_REPORT_PATH", str(isolated_pdf))
    st.cache_resource.clear()

    app_entry = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
    try:
        at = AppTest.from_file(app_entry, default_timeout=60).run()
        assert not at.exception

        # Quick-start button jump from Overview -> Image Analysis
        at.radio[0].set_value("🏠 Overview").run()
        assert not at.exception
        img_jump = [b for b in at.button if "Open Image Analysis" in b.label][0]
        img_jump.click().run()
        assert not at.exception
        assert at.radio[0].value == "🖼️ Image Analysis"

        # Switch to Benchmark Gallery, run real forensic exam & compile PDF in isolated DB/path
        if Path("data/demo_samples").exists() and list(Path("data/demo_samples").glob("*.jpg")):
            input_radio = [r for r in at.radio if r.key == "img_input_mode_selector"][0]
            input_radio.set_value("🧪 Preloaded Benchmark Gallery").run()
            assert not at.exception

            run_btn = [b for b in at.button if b.key == "run_img_exam_btn"][0]
            run_btn.click().run()
            assert not at.exception

            pdf_btn = [b for b in at.button if b.key == "gen_pdf"][0]
            pdf_btn.click().run()
            assert not at.exception
            assert isolated_pdf.exists()
            assert isolated_pdf.read_bytes().startswith(b"%PDF")

        # Navigate across remaining workspace sections cleanly
        for section in (
            "🎥 Video Analysis",
            "📊 Model Performance",
            "🔬 Forensic Evidence",
            "ℹ️ Model Information",
        ):
            at.radio[0].set_value(section).run()
            assert not at.exception, f"UI exception in {section}: {at.exception}"

        # Confirm isolated DB recorded examination + immutable upload/processing/prediction/pdf events
        db_check = ForensicDatabase(db_path=str(isolated_db))
        stats = db_check.get_summary_stats()
        assert stats["images"] >= 1
        assert db_check.get_schema_info()["total_audit_events"] >= 4
    finally:
        st.cache_resource.clear()


def test_production_integrity_guards():
    if CKPT_PATH.exists():
        digest = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
        assert digest == BASELINE_CKPT_SHA256

    prod_db = Path("data/forensic_history.db")
    if prod_db.exists():
        conn = sqlite3.connect(str(prod_db))
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(forensic_examinations)").fetchall()}
            trigs = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
            }
        finally:
            conn.close()
        assert {"evidence_id", "raw_file_sha256", "decision_threshold", "is_archived"}.issubset(cols)
        assert {"trg_audit_events_no_update", "trg_audit_events_no_delete"}.issubset(trigs)
