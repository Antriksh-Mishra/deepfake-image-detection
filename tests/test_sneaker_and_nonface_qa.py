"""
Unit & Integration Tests for Sneaker / Non-Face Media Handling,
Object-Aware Grad-CAM, Format-Aware ELA, and Invariant Verdict Coherence.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.core.face_extractor import FaceExtractor
from src.evaluation.calibration import (
    OUTCOME_AUTHENTIC,
    OUTCOME_FAKE,
    OUTCOME_MANIPULATED,
    OUTCOME_REAL,
    OUTCOME_UNCERTAIN,
    classify_with_policy,
)
from src.explainability.ela import ErrorLevelAnalysis
from src.explainability.gradcam import compute_regional_scores
from src.explainability.srm_analysis import SpatialSRMAnalyzer
from src.pipeline import ForensicInferencePipeline


def test_sneaker_image_classified_as_nonface_object():
    """Verifies that non-face object images (such as shoes/sneakers) are localized without facial assumptions."""
    extractor = FaceExtractor()
    # Create synthetic sneaker image: white studio canvas with colored shoe silhouette
    w, h = 300, 300
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 248
    canvas[160:220, 60:240] = [30, 90, 200]  # Sole
    canvas[100:160, 90:210] = [220, 40, 50]  # Upper
    img = Image.fromarray(canvas)

    all_faces = extractor.extract_all_faces(img)
    assert len(all_faces) == 1
    rec = all_faces[0]
    assert rec["has_face"] is False
    assert rec["face_detected"] is False
    assert "object_bbox" in rec
    assert "Face" not in rec["image_type"]
    assert "Product" in rec["image_type"] or "Object" in rec["image_type"]


def test_gradcam_nonface_regional_scores_has_no_facial_terms():
    """Ensures that Grad-CAM regional telemetry for non-face images uses object geometry, not nasolabial terms."""
    cam_dummy = np.random.rand(256, 256).astype(np.float32)

    # Face image -> Facial landmarks
    face_scores = compute_regional_scores(cam_dummy, has_face=True)
    assert "Mid-Facial / Nasolabial Region" in face_scores["region_scores"]
    assert face_scores["is_facial_context"] is True

    # Non-face sneaker / object image -> Structural zones
    obj_scores = compute_regional_scores(cam_dummy, has_face=False, image_type="Product / Commercial Object")
    assert "Mid-Facial / Nasolabial Region" not in obj_scores["region_scores"]
    assert "Nasolabial" not in obj_scores["dominant_region"]
    assert "Central Object Core / Foreground" in obj_scores["region_scores"]
    assert "Upper Structural Perimeter" in obj_scores["region_scores"]
    assert obj_scores["is_facial_context"] is False


def test_verdict_invariants_70pct_fake_is_never_real():
    """
    CRITICAL: Verifies that an AI-generated image with P(Manipulated)=70.25%
    can NEVER be classified as REAL / AUTHENTIC under any threshold.
    """
    # 1. Balanced threshold tau=0.50 with band [0.42, 0.58]
    policy_balanced = {
        "decision_threshold": 0.50,
        "uncertain_low": 0.42,
        "uncertain_high": 0.58,
    }
    res_50 = classify_with_policy(0.7025, policy=policy_balanced)
    assert res_50["prediction_outcome"] in (OUTCOME_FAKE, OUTCOME_MANIPULATED)
    assert res_50["is_fake"] is True
    assert res_50["is_uncertain"] is False

    # 2. Strict threshold tau=0.71 with band [0.707, 0.719]
    policy_strict = {
        "decision_threshold": 0.71,
        "uncertain_low": 0.707,
        "uncertain_high": 0.7188,
    }
    res_71 = classify_with_policy(0.7025, policy=policy_strict)
    # Must be UNCERTAIN — NEVER REAL / AUTHENTIC!
    assert res_71["prediction_outcome"] == OUTCOME_UNCERTAIN
    assert res_71["prediction_outcome"] != OUTCOME_REAL
    assert res_71["prediction_outcome"] != OUTCOME_AUTHENTIC
    assert res_71["is_uncertain"] is True


def test_ela_png_format_awareness():
    """Verifies that ELA correctly identifies lossless PNG containers and provides accurate caveats."""
    ela = ErrorLevelAnalysis(quality=90)
    img_png = Image.new("RGB", (128, 128), color=(200, 200, 200))
    res = ela.compute_ela_detailed(img_png, container_format="PNG", object_bbox=(20, 20, 80, 80))

    assert res["container_format"] == "PNG"
    assert res["is_lossless_source"] is True
    assert res["inconsistency_level"] == "Lossless Source Quantization"
    assert "lossless" in res["inconsistency_summary"].lower()
    assert res["foreground_mean_error"] is not None


def test_spatial_srm_analyzer():
    """Verifies that SpatialSRMAnalyzer computes non-trivial noise residuals and statistical metrics."""
    srm = SpatialSRMAnalyzer()
    img = Image.new("RGB", (128, 128), color=(120, 140, 160))
    res = srm.analyze_residuals(img)

    assert "mean_energy" in res
    assert "block_inconsistency_cv" in res
    assert "residual_kurtosis" in res
    assert "consistency_tier" in res
    assert res["heatmap_image"].size == (128, 128)
    assert res["mean_energy"] >= 0.0


def test_pipeline_latency_sum_parity():
    """Verifies that latency breakdown stages sum up to total measured latency without gaps."""
    pipe = ForensicInferencePipeline(device="cpu")
    dummy = Image.new("RGB", (200, 200), color=(100, 110, 120))
    res = pipe.analyze_image(dummy)

    lat = res["latency_breakdown"]
    total = lat["total_ms"]
    stages_sum = lat["preprocessing_ms"] + lat["neural_inference_ms"] + lat["explainability_ms"] + lat["postprocessing_ms"]
    assert abs(total - stages_sum) <= 0.2  # Rounding tolerance <= 0.2 ms
    assert "images/second" in res["performance_telemetry"]["throughput_label"]
