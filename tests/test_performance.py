"""
Tests for Phase 10 Performance Optimization:
- Process/accelerator memory telemetry & throughput computation
- Automatic accelerator-to-CPU RuntimeError fallback (safe_batched_forward)
- Batched vs single-sample numerical logit parity in pipeline, evaluate_split & RobustnessBenchmark
- Immediate mtime+size Streamlit @st.cache_data invalidation (B10 fix)
- Video analyzer performance_telemetry reporting
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from app.performance_dashboard import load_evaluation_data
from src.core.utils import (
    compute_throughput_fps,
    get_device,
    get_runtime_memory_telemetry,
    safe_batched_forward,
)
from src.evaluation.evaluate_checkpoint import evaluate_split
from src.evaluation.robustness_bench import RobustnessBenchmark
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet
from src.pipeline import ForensicInferencePipeline
from src.video.video_analyzer import VideoForensicAnalyzer


def test_runtime_memory_telemetry_and_throughput():
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"

    tel = get_runtime_memory_telemetry(cpu_dev)
    assert tel["device"] == "cpu"
    assert tel["cpu_rss_mb"] > 0.0
    assert tel["accelerator_alloc_mb"] == 0.0
    assert tel["accelerator_reserved_mb"] == 0.0
    assert tel["memory_backend"] in {"psutil", "resource.getrusage_peak_rss"}

    assert compute_throughput_fps(10, 500.0) == 20.0
    assert compute_throughput_fps(0, 500.0) == 0.0


def test_safe_batched_forward_cpu_fallback_on_runtime_error():
    class _FailingAccelModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def to(self, *args, **kwargs):
            # Ignore simulated pseudo-device transfers while keeping real CPU transfer clean
            return self

        def forward(self, sp, fr):
            if sp.device.type != "cpu":
                raise RuntimeError("Simulated MPS out-of-memory error")
            return torch.full((sp.shape[0], 1), 0.42, device=sp.device)

    model = _FailingAccelModel()
    sp = torch.zeros((3, 3, 16, 16))
    fr = torch.zeros((3, 1, 16, 16))

    # Simulate non-CPU execution failing cleanly and falling back to CPU
    class _FakeMetaTensorDevice:
        type = "mps"

    # Use real meta/tensor wrapper: subclass torch.Tensor is heavy; instead mock `.to` on inputs for deterministic test
    class _AccelTensorMock:
        def __init__(self, inner: torch.Tensor):
            self._inner = inner

        def to(self, dev):
            if getattr(dev, "type", str(dev)) == "cpu":
                return self._inner
            # Return tensor proxy whose `.device.type` reports "mps" while staying runnable
            class _Proxy:
                def __init__(self, d):
                    self.device = type("Dev", (), {"type": "mps"})()
                    self.shape = d.shape
            return _Proxy(self._inner)

    logits, active_dev, fallback = safe_batched_forward(
        model,
        _AccelTensorMock(sp),  # type: ignore[arg-type]
        _AccelTensorMock(fr),  # type: ignore[arg-type]
        _FakeMetaTensorDevice(),  # type: ignore[arg-type]
        enable_cpu_fallback=True,
    )
    assert fallback is True
    assert active_dev.type == "cpu"
    assert logits.shape == (3, 1)
    assert np.allclose(logits.detach().cpu().numpy(), 0.42)


def test_pipeline_multiface_batched_inference_parity():
    p1 = Path("data/demo_samples/authentic_studio_photo_1.jpg")
    p2 = Path("data/demo_samples/deepfake_stylegan_checkerboard.jpg")
    if not (p1.exists() and p2.exists()):
        return

    pipe = ForensicInferencePipeline(
        checkpoint_path="checkpoints/best_model.pth" if Path("checkpoints/best_model.pth").exists() else None,
        device="cpu",
    )
    f1 = Image.open(p1).convert("RGB").resize((240, 240))
    f2 = Image.open(p2).convert("RGB").resize((240, 240))
    canvas = Image.new("RGB", (680, 360), color=(35, 55, 80))
    canvas.paste(f1, (50, 60))
    canvas.paste(f2, (390, 60))

    res = pipe.analyze_image(canvas)
    tel = res["performance_telemetry"]
    assert res["faces_count"] == 2
    assert tel["batched_inference"] is True
    assert tel["batch_size"] == 2
    assert tel["faces_evaluated"] == 2
    assert tel["throughput_fps"] > 0.0
    assert tel["cpu_rss_mb"] > 0.0
    assert tel["cpu_fallback_triggered"] is False

    # Verify numerical parity between batched forward and sequential per-face forward
    all_faces = pipe.face_extractor.extract_all_faces(canvas)
    seq_reports = [
        pipe._analyze_face_record(f_rec, pipe.uncertainty_policy, canvas.size, include_ela_comparison=(i == 0))
        for i, f_rec in enumerate(all_faces)
    ]
    batched_probs = [r["fake_probability"] for r in res["multi_face_reports"]]
    seq_probs = [r["fake_probability"] for r in seq_reports]
    assert np.allclose(batched_probs, seq_probs, atol=1e-5)


def test_batched_evaluate_split_and_robustness_parity(tmp_path: Path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    imgs = []
    real_paths = []
    fake_paths = []
    for i in range(3):
        arr = np.full((64, 64, 3), 40 + i * 35, dtype=np.uint8)
        im = Image.fromarray(arr)
        imgs.append(im)
        rp = real_dir / f"r_{i}.jpg"
        fp = fake_dir / f"f_{i}.jpg"
        im.save(rp, format="JPEG", quality=92)
        im.save(fp, format="JPEG", quality=92)
        if i < 2:
            real_paths.append(rp)
        else:
            fake_paths.append(fp)

    corrupt_path = fake_dir / "corrupt.jpg"
    corrupt_path.write_bytes(b"not-a-valid-jpeg-stream")
    fake_paths.append(corrupt_path)

    model = DualDomainSentinelNet(pretrained=False).eval()
    analyzer = SpectralAnalyzer(target_size=256)
    dev = torch.device("cpu")

    m_single, y_t1, scores_single = evaluate_split(model, analyzer, real_paths, fake_paths, dev, batch_size=1)
    m_batch, y_t2, scores_batch = evaluate_split(model, analyzer, real_paths, fake_paths, dev, batch_size=3)

    assert np.array_equal(y_t1, y_t2)
    assert np.allclose(scores_single, scores_batch, atol=1e-5)
    assert scores_batch[-1] == 0.5  # Corrupt file safely assigned fallback without failing batch
    assert m_batch["sample_count"] == 4
    assert m_batch["throughput_fps"] > 0.0

    # RobustnessBenchmark batched vs single sample parity
    bench = RobustnessBenchmark(model=model, device=dev)
    probs_single = [bench.evaluate_sample(im) for im in imgs]
    probs_batch = bench.evaluate_batch(imgs, batch_size=2)
    assert np.allclose(probs_single, probs_batch, atol=1e-5)

    jpeg_accs = bench.run_jpeg_sweep(imgs, [0, 1, 0], qualities=[90, 50], batch_size=2)
    assert set(jpeg_accs.keys()) == {90, 50}


def test_streamlit_cache_invalidates_immediately_on_file_update_b10(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    cal_dir = res_dir / "calibration"
    ext_dir = res_dir / "external_benchmark"
    ckpt_dir.mkdir()
    cal_dir.mkdir(parents=True)
    ext_dir.mkdir(parents=True)

    metrics_file = ckpt_dir / "evaluation_metrics.json"
    metrics_file.write_text(json.dumps({"accuracy": 0.80, "sample_count": 100}))
    (ckpt_dir / "training_history.json").write_text(json.dumps({"train_loss": [0.5, 0.3]}))

    data_v1 = load_evaluation_data(repo_root=tmp_path)
    assert data_v1["metrics"]["accuracy"] == 0.80
    assert data_v1["metrics"]["sample_count"] == 100

    # Modify evaluation_metrics.json synchronously within milliseconds (< 60s old TTL)
    # Bump mtime cleanly by +2 seconds in case filesystem coarse resolution matches
    metrics_file.write_text(json.dumps({"accuracy": 0.91, "sample_count": 200}))
    stat_now = metrics_file.stat()
    os.utime(metrics_file, (stat_now.st_atime + 2, stat_now.st_mtime + 2))

    data_v2 = load_evaluation_data(repo_root=tmp_path)
    assert data_v2["metrics"]["accuracy"] == 0.91
    assert data_v2["metrics"]["sample_count"] == 200


def test_video_analyzer_includes_performance_telemetry(tmp_path: Path):
    vid_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (256, 256))
    face_p = Path("data/demo_samples/authentic_studio_photo_1.jpg")
    frame_rgb = (
        np.array(Image.open(face_p).convert("RGB").resize((256, 256)))
        if face_p.exists()
        else np.full((256, 256, 3), 120, dtype=np.uint8)
    )
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for _ in range(5):
        writer.write(frame_bgr)
    writer.release()

    analyzer = VideoForensicAnalyzer(checkpoint_path=None, device="cpu")
    res = analyzer.analyze_video(
        vid_path,
        sample_mode="uniform",
        max_frames=3,
        batch_size=2,
        allow_center_crop_fallback=True,
    )
    assert res["status"] != "error"
    tel = res["performance_telemetry"]
    assert tel["device"] == "cpu"
    assert tel["batched_inference"] is True
    assert tel["batch_size"] == 2
    assert tel["cpu_rss_mb"] > 0.0
    assert tel["processing_fps"] > 0.0
    assert "cpu_fallback_triggered" in tel
