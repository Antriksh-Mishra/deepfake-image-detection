"""
Unit & Integration Tests for Phase 5: Confidence Calibration, Threshold Optimization,
Uncertainty Handling (3-Outcome Policy), and Decoupled Pipeline Prediction.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation.calibration import (
    OUTCOME_AUTHENTIC,
    OUTCOME_MANIPULATED,
    OUTCOME_UNCERTAIN,
    analyze_thresholds_on_validation,
    classify_with_policy,
    compute_brier_score,
    compute_ece,
    fit_temperature_scaling,
    fit_uncertainty_policy_on_validation,
    run_calibration_from_logits,
    sigmoid_np,
)
from src.pipeline import ForensicInferencePipeline


def test_temperature_scaling_reduces_overconfident_val_nll():
    rng = np.random.RandomState(42)
    # Construct deliberately overconfident logits (high temperature T* > 2 improves NLL)
    y_val = np.array([0] * 100 + [1] * 100)
    base = np.concatenate([rng.normal(-1.0, 1.25, 100), rng.normal(1.0, 1.25, 100)])
    overconfident_logits = base * 4.5

    res = fit_temperature_scaling(overconfident_logits, y_val)
    assert res["fitted_on_split"] == "validation_only"
    assert res["temperature"] > 1.2
    assert res["val_nll_after"] < res["val_nll_before"]


def test_ece_and_brier_computation():
    y = np.array([0, 0, 1, 1])
    p_good = np.array([0.05, 0.10, 0.90, 0.95])
    p_bad = np.array([0.90, 0.85, 0.15, 0.10])

    ece_good = compute_ece(y, p_good, n_bins=5, mode="positive_class")
    ece_bad = compute_ece(y, p_bad, n_bins=5, mode="positive_class")
    ece_conf = compute_ece(y, p_good, n_bins=5, mode="confidence", decision_threshold=0.50)

    assert ece_good["ece"] < ece_bad["ece"]
    assert ece_conf["ece"] <= 0.10
    assert compute_brier_score(y, p_good) < compute_brier_score(y, p_bad)


def test_validation_threshold_and_pr_analysis():
    rng = np.random.RandomState(7)
    y_val = np.array([0] * 80 + [1] * 80)
    # Shifted probabilities where optimal separation sits around 0.65 rather than 0.50
    probs = np.concatenate([rng.uniform(0.25, 0.60, 80), rng.uniform(0.66, 0.95, 80)])

    th = analyze_thresholds_on_validation(y_val, probs, strategy="youden_j")
    assert th["fitted_on_split"] == "validation_only"
    assert th["default_threshold"] == 0.50
    assert 0.55 <= th["selected_threshold"] <= 0.72
    assert len(th["pr_curve_summary"]["precision"]) > 0
    assert len(th["pr_curve_summary"]["recall"]) == len(th["pr_curve_summary"]["precision"])


def test_three_outcome_policy_and_quality_guard():
    policy = {
        "decision_threshold": 0.50,
        "uncertain_low": 0.38,
        "uncertain_high": 0.62,
    }
    res_auth = classify_with_policy(0.18, policy=policy, quality_ok=True)
    res_manip = classify_with_policy(0.84, policy=policy, quality_ok=True)
    res_uncert = classify_with_policy(0.51, policy=policy, quality_ok=True)
    res_lowq = classify_with_policy(0.92, policy=policy, quality_ok=False, quality_reason="Too blurry")

    assert res_auth["prediction_outcome"] == OUTCOME_AUTHENTIC
    assert res_auth["is_fake"] is False and res_auth["is_uncertain"] is False

    assert res_manip["prediction_outcome"] == OUTCOME_MANIPULATED
    assert res_manip["is_fake"] is True and res_manip["is_uncertain"] is False

    assert res_uncert["prediction_outcome"] == OUTCOME_UNCERTAIN
    assert res_uncert["is_uncertain"] is True

    assert res_lowq["prediction_outcome"] == OUTCOME_UNCERTAIN
    assert res_lowq["is_uncertain"] is True
    assert "blurry" in res_lowq["uncertainty_reason"].lower()


def test_run_calibration_from_logits_generates_plots_and_json(tmp_path):
    rng = np.random.RandomState(123)
    y_val = np.array([0] * 60 + [1] * 60)
    val_logits = np.concatenate([rng.normal(-1.8, 1.1, 60), rng.normal(1.8, 1.1, 60)]) * 2.5

    y_test = np.array([0] * 50 + [1] * 50)
    test_logits = np.concatenate([rng.normal(-1.6, 1.1, 50), rng.normal(1.6, 1.1, 50)]) * 2.5

    out_dir = tmp_path / "cal_out"
    payload = run_calibration_from_logits(
        val_logits=val_logits,
        val_labels=y_val,
        test_logits=test_logits,
        test_labels=y_test,
        output_dir=str(out_dir),
    )

    assert (out_dir / "calibration_results.json").exists()
    assert Path(payload["artifacts"]["reliability_diagram"]).exists()
    assert Path(payload["artifacts"]["confidence_distribution"]).exists()
    assert Path(payload["artifacts"]["pr_threshold_analysis"]).exists()
    assert payload["protocol_guarantees"]["zero_test_set_tuning"] is True
    assert payload["protocol_guarantees"]["bayesian_heuristic_override_removed"] is True
    assert set(payload["outcomes"]) == {OUTCOME_AUTHENTIC, OUTCOME_MANIPULATED, OUTCOME_UNCERTAIN}
    assert payload["test"]["calibrated"]["brier_score"] <= payload["test"]["uncalibrated"]["brier_score"] + 1e-6


def test_pipeline_uses_decoupled_calibrated_classifier(tmp_path):
    cal_file = tmp_path / "calibration_results.json"
    cal_file.write_text(
        json.dumps(
            {
                "temperature_scaling": {"temperature": 2.0},
                "uncertainty_policy": {
                    "decision_threshold": 0.50,
                    "uncertain_low": 0.40,
                    "uncertain_high": 0.60,
                    "min_laplacian_sharpness": 15.0,
                    "min_dimension_px": 80,
                },
            }
        )
    )

    pipe = ForensicInferencePipeline(device="cpu", calibration_path=str(cal_file))
    # Generate textured 256x256 RGB image (passes sharpness threshold so calibration band controls outcome)
    rng = np.random.RandomState(5)
    arr = rng.randint(40, 210, size=(256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    res = pipe.analyze_image(img)
    cal_meta = res["calibration_metadata"]
    ev = res["forensic_evidence"]

    assert cal_meta["calibrated"] is True
    assert cal_meta["temperature"] == pytest.approx(2.0)
    # Primary fake_probability MUST match calibrated neural output with ZERO spectral/ELA override
    assert res["fake_probability"] == pytest.approx(cal_meta["calibrated_prob"], abs=1e-6)
    assert res["fake_probability"] == pytest.approx(ev["neural_fake_prob"], abs=1e-6)
    assert ev["evidence_decoupled_from_classifier"] is True
    assert res["prediction_outcome"] in {OUTCOME_AUTHENTIC, OUTCOME_MANIPULATED, OUTCOME_UNCERTAIN}


def test_bounded_logits_uncertainty_policy_is_non_degenerate():
    """Ensures bounded/compressed calibrated probs never swallow MANIPULATED or AUTHENTIC into 0 predictions."""
    rng = np.random.RandomState(11)
    y_val = np.array([0] * 100 + [1] * 100)
    # Tightly upper-skewed bounded probability scenario [0.29, 0.72] like LabelSmoothing + high T
    val_probs = np.concatenate([
        rng.uniform(0.29, 0.715, 100),
        np.concatenate([rng.uniform(0.29, 0.31, 25), rng.uniform(0.710, 0.722, 75)]),
    ])
    th = analyze_thresholds_on_validation(y_val, val_probs, strategy="youden_j")
    pol = fit_uncertainty_policy_on_validation(y_val, val_probs, decision_threshold=th["selected_threshold"])
    n_auth = int(np.sum(val_probs < pol["uncertain_low"]))
    n_manip = int(np.sum(val_probs > pol["uncertain_high"]))
    n_unc = int(np.sum((val_probs >= pol["uncertain_low"]) & (val_probs <= pol["uncertain_high"])))
    assert pol["uncertain_low"] < pol["decision_threshold"] < pol["uncertain_high"]
    assert n_auth > 0
    assert n_manip > 0
    assert n_unc > 0


def test_real_calibration_results_artifact_exists():
    cal_path = Path("results/calibration/calibration_results.json")
    assert cal_path.exists(), "Run `python -m src.evaluation.calibration` to generate calibration_results.json"
    data = json.loads(cal_path.read_text())
    assert data["protocol_guarantees"]["zero_test_set_tuning"] is True
    assert data["temperature_scaling"]["val_nll_after"] <= data["temperature_scaling"]["val_nll_before"]
    for split_key in ("validation", "test"):
        eval_3 = data[split_key]["calibrated"]["three_outcome_policy_eval"]
        assert eval_3["authentic_count"] > 0
        assert eval_3["manipulated_count"] > 0
        assert eval_3["uncertain_count"] > 0
    for art in ("reliability_diagram", "confidence_distribution", "pr_threshold_analysis"):
        assert Path(data["artifacts"][art]).exists()
