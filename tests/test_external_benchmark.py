"""
Unit & Integration Tests for Phase 3 External Dataset Loaders
(FaceForensics++, Celeb-DF, DFDC) and Independent Generalization Benchmark Runner.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.external_datasets import (
    load_celebdf_samples,
    load_dfdc_samples,
    load_faceforensics_samples,
)
from src.evaluation.external_benchmark import run_external_benchmarks
from src.evaluation.metrics import evaluate_predictions, plot_precision_recall_curve


def _write_dummy_rgb_image(path: Path, color: tuple, size=(128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, quality=90)


def _write_dummy_video(path: Path, color_bgr: tuple, n_frames: int = 6, size=(128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    assert writer.isOpened()
    frame = np.full((size[1], size[0], 3), color_bgr, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def test_faceforensics_loader_and_compression_filter(tmp_path):
    ff_root = tmp_path / "FaceForensics++"
    _write_dummy_rgb_image(ff_root / "original_sequences/youtube/c23/images/000_f0.png", (180, 140, 120))
    _write_dummy_rgb_image(ff_root / "original_sequences/youtube/c40/images/000_f0.png", (175, 135, 115))
    _write_dummy_rgb_image(ff_root / "manipulated_sequences/Deepfakes/c23/images/000_003_f0.png", (120, 160, 180))
    _write_dummy_rgb_image(ff_root / "manipulated_sequences/FaceSwap/c40/images/001_002_f0.png", (110, 150, 170))

    info_all = load_faceforensics_samples(ff_root, compression="all")
    assert info_all.available is True
    assert info_all.real_count == 2
    assert info_all.fake_count == 2
    assert info_all.sub_categories["original"] == 2
    assert info_all.sub_categories["Deepfakes"] == 1
    assert info_all.sub_categories["FaceSwap"] == 1

    info_c23 = load_faceforensics_samples(ff_root, compression="c23")
    assert info_c23.sample_count == 2
    assert info_c23.real_count == 1
    assert info_c23.fake_count == 1


def test_celebdf_loader_inverted_txt_convention_and_video(tmp_path):
    celeb_root = tmp_path / "Celeb-DF"
    _write_dummy_video(celeb_root / "Celeb-real/id0_0000.avi", (140, 150, 190))
    _write_dummy_video(celeb_root / "YouTube-real/id1_0001.avi", (130, 140, 180))
    _write_dummy_video(celeb_root / "Celeb-synthesis/id0_id1_0000.avi", (90, 180, 110))

    # Official Celeb-DF list uses 1=Real, 0=Fake; verify mapping normalizes to our 0=Real, 1=Fake
    (celeb_root / "List_of_testing_videos.txt").write_text(
        "1 Celeb-real/id0_0000.avi\n"
        "1 YouTube-real/id1_0001.avi\n"
        "0 Celeb-synthesis/id0_id1_0000.avi\n"
    )
    info = load_celebdf_samples(celeb_root, use_official_test_list=True)
    assert info.available is True
    assert info.real_count == 2
    assert info.fake_count == 1
    assert all(s.media_type == "video" for s in info.samples)
    real_labels = [s.label for s in info.samples if "real" in s.sub_category.lower()]
    fake_labels = [s.label for s in info.samples if "synthesis" in s.sub_category.lower()]
    assert real_labels == [0, 0]
    assert fake_labels == [1]


def test_dfdc_loader_metadata_and_csv(tmp_path):
    dfdc_root = tmp_path / "DFDC"
    _write_dummy_rgb_image(dfdc_root / "dfdc_train_part_0/clip_a.jpg", (200, 180, 160))
    _write_dummy_rgb_image(dfdc_root / "dfdc_train_part_0/clip_b.jpg", (80, 100, 210))
    (dfdc_root / "dfdc_train_part_0/metadata.json").write_text(
        json.dumps({
            "clip_a.jpg": {"label": "REAL", "original": "clip_a.jpg"},
            "clip_b.jpg": {"label": "FAKE", "original": "clip_a.jpg"},
        })
    )
    _write_dummy_rgb_image(dfdc_root / "test_videos/clip_c.png", (100, 220, 150))
    (dfdc_root / "labels.csv").write_text("filename,label\nclip_c.png,1\n")

    info = load_dfdc_samples(dfdc_root)
    assert info.available is True
    assert info.sample_count == 3
    assert info.real_count == 1
    assert info.fake_count == 2


def test_metrics_fpr_fnr_pr_curve(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.7, 0.2, 0.9])
    m = evaluate_predictions(y_true, y_scores, threshold=0.5)
    assert m["fpr"] == pytest.approx(0.5)
    assert m["fnr"] == pytest.approx(0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["pr_auc"] is not None

    pr_out = tmp_path / "pr.png"
    plot_precision_recall_curve(y_true, y_scores, save_path=str(pr_out))
    assert pr_out.exists() and pr_out.stat().st_size > 1000


def test_independent_external_benchmark_end_to_end_and_unavailable(tmp_path, monkeypatch):
    # Unavailable check using clean isolated env vars
    monkeypatch.setenv("FFPP_DIR", str(tmp_path / "missing_ffpp"))
    monkeypatch.setenv("CELEBDF_DIR", str(tmp_path / "missing_celebdf"))
    monkeypatch.setenv("DFDC_DIR", str(tmp_path / "missing_dfdc"))

    unavail_out = tmp_path / "ext_unavail"
    summary_unavail = run_external_benchmarks(
        checkpoint_path="checkpoints/best_model.pth",
        output_dir=str(unavail_out),
        device_str="cpu",
    )
    assert summary_unavail["available_dataset_count"] == 0
    for k in ("faceforensics", "celeb_df", "dfdc"):
        assert summary_unavail["datasets"][k]["available"] is False
        assert summary_unavail["datasets"][k]["metrics"] is None

    # Available FF++ (images) + Celeb-DF (AVI videos) using existing checkpoint without training/tuning
    ff_dir = tmp_path / "FFPP"
    _write_dummy_rgb_image(ff_dir / "original_sequences/youtube/c23/images/000_0.jpg", (210, 175, 150))
    _write_dummy_rgb_image(ff_dir / "original_sequences/youtube/c23/images/001_0.jpg", (200, 170, 145))
    _write_dummy_rgb_image(ff_dir / "manipulated_sequences/Deepfakes/c23/images/000_001_0.jpg", (90, 210, 130))
    _write_dummy_rgb_image(ff_dir / "manipulated_sequences/FaceSwap/c23/images/001_000_0.jpg", (70, 120, 230))

    celeb_dir = tmp_path / "CelebDF"
    _write_dummy_video(celeb_dir / "Celeb-real/id0_000.avi", (160, 170, 210), n_frames=4)
    _write_dummy_video(celeb_dir / "Celeb-synthesis/id0_id1_000.avi", (110, 200, 100), n_frames=4)

    monkeypatch.setenv("FFPP_DIR", str(ff_dir))
    monkeypatch.setenv("CELEBDF_DIR", str(celeb_dir))

    eval_out = tmp_path / "ext_eval"
    summary_eval = run_external_benchmarks(
        checkpoint_path="checkpoints/best_model.pth",
        output_dir=str(eval_out),
        threshold=0.50,
        frames_per_video=3,
        align_faces=False,
        device_str="cpu",
    )
    assert summary_eval["available_dataset_count"] == 2
    assert summary_eval["evaluation_guarantees"]["no_training_on_external_data"] is True
    assert summary_eval["evaluation_guarantees"]["no_test_set_threshold_tuning"] is True

    ff_res = summary_eval["datasets"]["faceforensics"]
    assert ff_res["status"] == "evaluated"
    assert ff_res["metrics"]["sample_count"] == 4
    assert ff_res["metrics"]["threshold_used"] == 0.50
    for required_key in ("accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc", "fpr", "fnr", "eer", "confusion_matrix"):
        assert required_key in ff_res["metrics"]
    for art_key in ("confusion_matrix", "roc_curve", "pr_curve"):
        assert Path(ff_res["artifacts"][art_key]).exists()

    cd_res = summary_eval["datasets"]["celeb_df"]
    assert cd_res["status"] == "evaluated"
    assert cd_res["metrics"]["frames_evaluated"] == 6
    assert summary_eval["datasets"]["dfdc"]["available"] is False
    assert summary_eval["datasets"]["dfdc"]["metrics"] is None


def test_real_external_benchmark_summary_honest():
    summary_file = Path("results/external_benchmark/external_benchmark_summary.json")
    assert summary_file.exists()
    data = json.loads(summary_file.read_text())
    assert data["evaluation_guarantees"]["never_fabricates_unavailable_results"] is True
    for k in ("faceforensics", "celeb_df", "dfdc"):
        assert k in data["datasets"]
        if not data["datasets"][k]["available"]:
            assert data["datasets"][k]["metrics"] is None
