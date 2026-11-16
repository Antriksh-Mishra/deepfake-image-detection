"""
Unit tests for Explainability Modules: ELA, 2D FFT Spectral Analysis, and Grad-CAM.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.explainability.ela import ELA_FORENSIC_LIMITATIONS, ErrorLevelAnalysis
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.explainability.gradcam import GradCAM, compute_regional_scores
from src.models.dual_domain_net import DualDomainSentinelNet
from src.models.spatial_backbone import SpatialBackbone
from src.pipeline import ForensicInferencePipeline


def test_ela_module():
    ela = ErrorLevelAnalysis(quality=90, scale=15)
    img = Image.new("RGB", (128, 128), color=(200, 100, 50))
    ela_img, mean_err = ela.compute_ela(img)
    assert ela_img.size == (128, 128)
    assert isinstance(mean_err, float)
    assert mean_err >= 0.0

    # Test detailed ELA + compression inconsistency + explicit limitations
    detailed = ela.compute_ela_detailed(img)
    assert "quadrant_errors" in detailed
    assert "hist_counts" in detailed
    assert "quad_discrepancy_ratio" in detailed
    assert "block_inconsistency_cv" in detailed
    assert "inconsistency_level" in detailed
    assert detailed["ela_heatmap_color"].size == (128, 128)
    assert len(detailed["quadrant_errors"]) == 4
    assert detailed["limitations"] == ELA_FORENSIC_LIMITATIONS and len(detailed["limitations"]) >= 4

    # Test multi-quality JPEG recompression comparison
    comps = ela.compute_recompression_comparison(img, qualities=(95, 90, 75))
    assert [c["quality"] for c in comps] == [95, 90, 75]
    assert all(c["ela_heatmap_color"].size == (128, 128) for c in comps)


def test_spectral_analyzer(tmp_path):
    analyzer = SpectralAnalyzer(target_size=128)
    # Build textured portrait crop so FFT spectrum is non-degenerate
    rng = np.random.RandomState(42)
    arr = rng.randint(40, 220, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    mag_spec, raw_log = analyzer.compute_fft_magnitude(img)

    assert mag_spec.shape == (128, 128)
    assert mag_spec.dtype == np.uint8

    cmap_img = analyzer.compute_colormap_log_spectrum(raw_log)
    assert cmap_img.size == (128, 128)

    freqs, profile = analyzer.azimuthal_average(raw_log)
    assert len(freqs) == len(profile)
    assert len(freqs) == 64  # radius for 128x128

    peak_res = analyzer.analyze_spectral_peaks(freqs, profile)
    assert "detected_peaks" in peak_res
    assert "high_freq_ratio" in peak_res
    assert "spectral_smoothness" in peak_res
    assert isinstance(peak_res["has_deconv_spikes"], bool)

    anom_img, bands = analyzer.compute_frequency_anomaly_map(raw_log, freqs, peak_res)
    assert anom_img.size == (128, 128)
    assert isinstance(bands, list)

    # Real vs suspicious comparison (using missing dir falls back gracefully without crashing)
    comp_fallback = analyzer.compare_real_vs_suspicious(freqs, profile, peak_res, reference_dir=str(tmp_path / "missing"))
    assert comp_fallback["available"] is False
    assert "comparison_summary" in comp_fallback

    # With actual demo_samples reference folder
    comp_real = analyzer.compare_real_vs_suspicious(freqs, profile, peak_res, reference_dir="data/demo_samples")
    if Path("data/demo_samples").exists():
        assert comp_real["available"] is True
        assert comp_real["reference_real_mean"] is not None


def test_gradcam():
    # Verify SpatialBackbone shape compatibility
    backbone = SpatialBackbone(backbone_name="efficientnet_b0", pretrained=False, feat_dim=1)
    cam_bb = GradCAM(model=backbone, target_layer=backbone.target_cam_layer)
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    hm_bb = cam_bb.generate_cam(x, target_class=0)
    assert hm_bb.shape == (256, 256)
    cam_bb.remove_hooks()

    # Verify B5 fix on non-vanishing DualDomainSentinelNet (non-zero single-logit grads; target_class 0 != 1)
    ckpt_path = Path("checkpoints/best_model.pth")
    if ckpt_path.exists():
        net = DualDomainSentinelNet(backbone_name="efficientnet_b0", pretrained=False)
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
        net.load_state_dict(sd, strict=True)
    else:
        net = DualDomainSentinelNet(backbone_name="resnet34", pretrained=False)

    cam_engine = GradCAM(model=net, target_layer=net.spatial_stream.target_cam_layer)
    spec = torch.randn(1, 1, 256, 256)

    hm_fake = cam_engine.generate_cam(x, spectral_tensor=spec, target_class=1)
    hm_real = cam_engine.generate_cam(x, spectral_tensor=spec, target_class=0)

    assert hm_fake.shape == (256, 256)
    assert hm_real.shape == (256, 256)
    assert hm_fake.min() >= 0.0 and hm_fake.max() <= 1.0 + 1e-5
    assert hm_real.min() >= 0.0 and hm_real.max() <= 1.0 + 1e-5
    assert hm_fake.max() > 0.0 and hm_real.max() > 0.0
    # Fixing B5: target_class=0 maximizes -logit rather than collapsing to +logit
    assert not np.allclose(hm_fake, hm_real, atol=1e-4)

    # Shape-mismatch safety in overlay_cam (e.g. 64x64 heatmap onto 200x200 RGB image)
    rgb_img = np.full((200, 200, 3), 128, dtype=np.uint8)
    small_hm = np.ones((64, 64), dtype=np.float32) * 0.5
    blended = cam_engine.overlay_cam(rgb_img, small_hm, alpha=0.45)
    assert blended.shape == (200, 200, 3) and blended.dtype == np.uint8

    reg = compute_regional_scores(hm_fake)
    assert reg["dominant_region"] in set(reg["region_scores"].keys())

    # Works even when model weights and input lack initial requires_grad
    for p in net.parameters():
        p.requires_grad_(False)
    hm_frozen = cam_engine.generate_cam(x.detach(), spectral_tensor=spec, target_class=1)
    assert hm_frozen.shape == (256, 256)

    cam_engine.remove_hooks()


def test_pipeline_unified_forensic_panel_outputs():
    pipe = ForensicInferencePipeline(device="cpu")
    sample_path = Path("data/demo_samples/authentic_studio_photo_1.jpg")
    img = Image.open(sample_path).convert("RGB") if sample_path.exists() else Image.new("RGB", (256, 256), color=(140, 120, 110))

    res = pipe.analyze_image(img)
    required_keys = [
        "prediction_outcome",
        "confidence_percent",
        "gradcam_overlay",
        "fft_spectrum_img",
        "fft_log_colormap_img",
        "fft_anomaly_map",
        "azimuthal_frequencies",
        "azimuthal_profile",
        "spectral_comparison",
        "ela_image",
        "ela_heatmap_color",
        "ela_comparison",
        "ela_limitations",
        "full_image_annotated",
        "forensic_observations",
    ]
    for k in required_keys:
        assert k in res, f"Missing unified forensic panel key: {k}"

    assert len(res["ela_comparison"]) == 3
    assert len(res["forensic_observations"]) >= 5
    assert any("definitive" in obs.lower() for obs in res["forensic_observations"])
