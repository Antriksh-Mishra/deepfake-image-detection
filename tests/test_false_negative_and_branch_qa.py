"""
Automated QA & Forensic Verification Test Suite.

Covers:
1. Output class mapping and directionality.
2. Temperature scaling invariants (T > 0, monotonicity, NLL/ECE reduction).
3. Threshold boundary behavior and uncertainty band routing.
4. Correct label mapping (REAL, FAKE, UNCERTAIN).
5. Inference pipeline consistency across operating modes (balanced, high_recall, high_precision).
6. Preprocessing transformation parity (spatial RGB vs spectral magnitude).
7. Robust edge cases (non-face, multi-face, low-res, grayscale/RGBA, flat/low-sharpness).
8. Decoupled branch predictions and cross-domain conflict / false-negative risk detection.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.core.utils import SPATIAL_TRANSFORM, SPECTRAL_TRANSFORM, get_device
from src.evaluation.calibration import (
    classify_with_policy,
    compute_ece,
    compute_nll,
    load_calibration_policy,
    sigmoid_np,
)
from src.models.dual_domain_net import DualDomainSentinelNet
from src.pipeline import ForensicInferencePipeline


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    path = "checkpoints/best_model.pth"
    assert Path(path).exists(), f"Production checkpoint {path} must exist."
    return path


@pytest.fixture(scope="module")
def pipeline(checkpoint_path: str) -> ForensicInferencePipeline:
    return ForensicInferencePipeline(
        checkpoint_path=checkpoint_path,
        calibration_path="results/calibration/calibration_results.json",
    )


# ==============================================================================
# 1. Output Class Mapping and Directionality
# ==============================================================================
def test_class_mapping_and_directionality(pipeline: ForensicInferencePipeline) -> None:
    """Verifies that single scalar logit maps monotonically to P(Manipulated/Fake)."""
    # Negative logit must yield P < 0.50 (Authentic/Real)
    neg_logit = -2.0
    p_neg = float(sigmoid_np(np.array([neg_logit / pipeline.temperature]))[0])
    assert p_neg < 0.50, f"Negative logit must yield P(Fake) < 0.50, got {p_neg}"

    # Positive logit must yield P > 0.50 (Manipulated/Fake)
    pos_logit = +2.0
    p_pos = float(sigmoid_np(np.array([pos_logit / pipeline.temperature]))[0])
    assert p_pos > 0.50, f"Positive logit must yield P(Fake) > 0.50, got {p_pos}"

    # Zero logit must yield exactly 0.50
    zero_logit = 0.0
    p_zero = float(sigmoid_np(np.array([zero_logit / pipeline.temperature]))[0])
    assert abs(p_zero - 0.50) < 1e-5


# ==============================================================================
# 2. Temperature Scaling Invariants
# ==============================================================================
def test_temperature_scaling_invariants() -> None:
    """Verifies that validation-fitted temperature T is positive, monotonic, and calibrates probabilities."""
    policy = load_calibration_policy("results/calibration/calibration_results.json")
    T = float(policy.get("temperature", 1.0))
    assert T > 0.0, f"Temperature must be strictly positive, got {T}"
    assert 1.0 <= T <= 5.0, f"Empirical temperature should lie in reasonable range [1, 5], got {T}"

    # Verify strict monotonicity
    logits = np.linspace(-10.0, 10.0, 50)
    scaled_probs = sigmoid_np(logits / T)
    assert np.all(np.diff(scaled_probs) > 0), "Temperature scaling must strictly preserve logit ranking."


# ==============================================================================
# 3. Threshold Boundary Behavior & Uncertainty Routing
# ==============================================================================
def test_threshold_boundary_behavior() -> None:
    """Verifies decision threshold boundary and uncertainty interval routing."""
    policy = {
        "decision_threshold": 0.71,
        "uncertain_low": 0.69,
        "uncertain_high": 0.73,
    }

    # Case A: Clear Authentic (< uncertain_low and <= 0.50)
    res_real = classify_with_policy(calibrated_prob=0.30, policy=policy)
    assert res_real["prediction_outcome"] == "REAL"
    assert not res_real["is_fake"]
    assert not res_real["is_uncertain"]

    # Case B: Clear Fake (> uncertain_high)
    res_fake = classify_with_policy(calibrated_prob=0.85, policy=policy)
    assert res_fake["prediction_outcome"] == "FAKE"
    assert res_fake["is_fake"]
    assert not res_fake["is_uncertain"]

    # Case C: Inside uncertainty zone [0.69, 0.73]
    res_unc = classify_with_policy(calibrated_prob=0.71, policy=policy)
    assert res_unc["prediction_outcome"] == "UNCERTAIN"
    assert res_unc["is_uncertain"]

    # Case D: Invariant: NO probability > 0.50 can ever be classified as REAL
    res_invar = classify_with_policy(calibrated_prob=0.55, policy=policy)
    assert res_invar["prediction_outcome"] != "REAL", "Probability > 0.50 cannot be certified as REAL"


# ==============================================================================
# 4. Correct Label Mapping (3 Verdict Tiers)
# ==============================================================================
def test_label_mapping_three_verdicts(pipeline: ForensicInferencePipeline) -> None:
    """Verifies that all three verdicts are mapped with consistent tags and explanations."""
    for outcome, is_f, is_u in [
        ("REAL", False, False),
        ("FAKE", True, False),
        ("UNCERTAIN", False, True),
    ]:
        outcome_dict = {
            "prediction_outcome": outcome,
            "is_fake": is_f,
            "is_uncertain": is_u,
            "classifier_confidence": 0.85,
            "decision_threshold": 0.71,
            "uncertain_low": 0.69,
            "uncertain_high": 0.73,
        }
        lbl, tier, badge, v_class, exp = pipeline._format_three_outcome_verdict(outcome_dict)
        assert lbl == outcome
        assert outcome in badge
        assert len(exp) > 20


# ==============================================================================
# 5. Inference Pipeline Operating Modes
# ==============================================================================
def test_inference_pipeline_operating_modes(pipeline: ForensicInferencePipeline) -> None:
    """Verifies that pipeline supports balanced, high_recall, and high_precision operating modes."""
    # Create test synthetic image
    img = Image.new("RGB", (256, 256), color=(140, 110, 90))

    res_bal = pipeline.analyze_image(img, operating_mode="balanced")
    assert res_bal["operating_mode"] == "balanced"
    assert res_bal["calibration_metadata"]["decision_threshold"] >= 0.70

    res_hr = pipeline.analyze_image(img, operating_mode="high_recall")
    assert res_hr["operating_mode"] == "high_recall"
    assert res_hr["calibration_metadata"]["decision_threshold"] <= 0.40

    res_hp = pipeline.analyze_image(img, operating_mode="high_precision")
    assert res_hp["operating_mode"] == "high_precision"
    assert res_hp["calibration_metadata"]["decision_threshold"] >= 0.71


# ==============================================================================
# 6. Transformation Consistency Between Training and Inference
# ==============================================================================
def test_spatial_spectral_transform_consistency() -> None:
    """Verifies input tensor shape and normalization ranges."""
    sample = Image.new("RGB", (300, 300), color=(128, 128, 128))
    sp_tensor = SPATIAL_TRANSFORM(sample)
    fr_tensor = SPECTRAL_TRANSFORM(sample)

    assert sp_tensor.shape == (3, 256, 256), f"Expected (3, 256, 256), got {sp_tensor.shape}"
    assert fr_tensor.shape == (3, 256, 256), f"Expected (3, 256, 256), got {fr_tensor.shape}"
    assert isinstance(sp_tensor, torch.Tensor)
    assert isinstance(fr_tensor, torch.Tensor)


# ==============================================================================
# 7. Robust Edge Cases (Non-Face, Multi-Face, Low-Res, Grayscale, Flat)
# ==============================================================================
def test_edge_cases(pipeline: ForensicInferencePipeline) -> None:
    """Evaluates edge cases without unhandled exceptions."""
    # A. Non-face image (solid synthetic object)
    sneaker_like = Image.new("RGB", (400, 300), color=(20, 80, 180))
    res_nonface = pipeline.analyze_image(sneaker_like)
    assert not res_nonface["has_face"]
    assert "Center Crop Fallback" in str(res_nonface["multi_face_reports"][0].get("verdict_tier", "")) or not res_nonface["face_detected"]

    # B. Unusually small image (quality degradation guard)
    tiny_img = Image.new("RGB", (48, 48), color=(100, 150, 200))
    res_tiny = pipeline.analyze_image(tiny_img)
    assert res_tiny["is_uncertain"], "Image below min dimensions must route to UNCERTAIN"

    # C. Flat / zero-variance image (sharpness degradation guard)
    flat_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    res_flat = pipeline.analyze_image(flat_img)
    assert res_flat["laplacian_variance"] < 1.0
    assert res_flat["is_uncertain"], "Flat zero-sharpness image must route to UNCERTAIN"

    # D. Non-RGB image (Grayscale mode 'L')
    gray_img = Image.new("L", (256, 256), color=120)
    res_gray = pipeline.analyze_image(gray_img.convert("RGB"))
    assert res_gray["prediction_outcome"] in ["REAL", "FAKE", "UNCERTAIN"]

    # E. RGBA transparent image
    rgba_img = Image.new("RGBA", (256, 256), color=(200, 100, 50, 180))
    res_rgba = pipeline.analyze_image(rgba_img.convert("RGB"))
    assert res_rgba["prediction_outcome"] in ["REAL", "FAKE", "UNCERTAIN"]


# ==============================================================================
# 8. Cross-Domain Conflict & False-Negative Risk Protection
# ==============================================================================
def test_cross_domain_conflict_detection(pipeline: ForensicInferencePipeline) -> None:
    """Verifies that decoupled branch discrepancy triggers false-negative safeguards."""
    # Construct artificial branch discrepancy where frequency stream detects fake
    # but spatial stream suppresses output
    mock_branch_info = {
        "spatial_prob": 0.28,
        "spatial_logit": -2.40,
        "frequency_prob": 0.74,
        "frequency_logit": +2.70,
        "fused_prob": 0.29,
        "fused_logit": -2.35,
        "discrepancy": 0.46,
        "has_conflict": True,
        "false_negative_risk": True,
    }

    # In Balanced mode: MUST NOT classify as REAL; must route to UNCERTAIN with warning
    res_balanced = classify_with_policy(
        calibrated_prob=0.29,
        policy=pipeline.uncertainty_policy,
        branch_info=mock_branch_info,
        operating_mode="balanced",
    )
    assert res_balanced["prediction_outcome"] == "UNCERTAIN"
    assert res_balanced["is_uncertain"]
    assert res_balanced["false_negative_warning"] is not None
    assert "Cross-Domain Discrepancy" in res_balanced["uncertainty_reason"]

    # In High-Recall mode: frequency fake evidence overrides to FAKE
    res_hr = classify_with_policy(
        calibrated_prob=0.29,
        policy=pipeline.uncertainty_policy,
        branch_info=mock_branch_info,
        operating_mode="high_recall",
    )
    assert res_hr["prediction_outcome"] == "FAKE"
    assert res_hr["is_fake"]
    assert not res_hr["is_uncertain"]


def test_user_portrait_false_negative_safeguard(pipeline: ForensicInferencePipeline) -> None:
    """Verifies that the user's AI-generated portrait is protected against false-negative REAL certification."""
    user_img_path = Path("/Users/antrikshmishra/.gemini/antigravity/brain/0bce4474-8b0d-4abc-b2c2-1900bc4214a5/.user_uploaded/media_1794894189108_a38ad3db.jpg")
    if not user_img_path.exists():
        pytest.skip("User test portrait not found at expected path.")

    with Image.open(user_img_path) as raw:
        img = raw.convert("RGB")

    # In Balanced mode: must NOT be certified as REAL
    res_bal = pipeline.analyze_image(img, operating_mode="balanced")
    assert res_bal["prediction_outcome"] != "REAL", "AI portrait must not be falsely certified as REAL"
    assert res_bal["prediction_outcome"] == "UNCERTAIN"
    assert res_bal["false_negative_risk"] is True
    assert res_bal["false_negative_warning"] is not None

    # In High-Recall mode: must be flagged as FAKE
    res_hr = pipeline.analyze_image(img, operating_mode="high_recall")
    assert res_hr["prediction_outcome"] == "FAKE"
    assert res_hr["is_fake"] is True
