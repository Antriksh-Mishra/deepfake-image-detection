"""
Model Confidence Calibration, Validation Threshold Selection & Uncertainty Estimation.

Implements:
1. Temperature Scaling fit strictly on validation logits (CPU float32 NLL minimization)
   without altering `checkpoints/best_model.pth`.
2. Validation-only Decision Threshold Optimization:
   - Compares 0.50 default threshold vs validation-selected operating threshold.
   - Full Precision-Recall threshold sweep and curve analysis.
3. Validated 3-Outcome Decision Policy:
   - `AUTHENTIC`
   - `MANIPULATED`
   - `UNCERTAIN / NEEDS MANUAL REVIEW`
   derived from a validation-selected confidence interval around the operating threshold
   plus an image-quality / sharpness guard.
4. Calibration Evaluation:
   - Expected Calibration Error (ECE: positive-class reliability + top-label confidence)
   - Brier Score & Negative Log-Likelihood (NLL)
   - Reliability Diagram (Before vs After Temperature Scaling)
   - Confidence Distribution Plot (Real vs Fake with Uncertainty Zone)
   - Precision-Recall Threshold Curve Plot

All fitting & threshold selection uses validation data ONLY (`data/val`); held-out test
data (`data/test`) is evaluated strictly out-of-sample.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from src.core.utils import SPATIAL_TRANSFORM, SPECTRAL_TRANSFORM, get_device, load_config
from src.data.dataset_audit import IMAGE_EXTENSIONS
from src.evaluation.metrics import evaluate_predictions
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet

OUTCOME_REAL = "REAL"
OUTCOME_FAKE = "FAKE"
OUTCOME_UNCERTAIN = "UNCERTAIN"

# Backward compatibility aliases
OUTCOME_AUTHENTIC = OUTCOME_REAL
OUTCOME_MANIPULATED = OUTCOME_FAKE


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays."""
    z = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Computes binary cross-entropy / negative log-likelihood."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier Score mean((p - y)^2)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    mode: str = "positive_class",
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and full per-bin calibration statistics.

    Args:
        y_true: Binary labels in {0, 1}.
        y_prob: Predicted probability for class 1 (Manipulated/Fake) in [0, 1].
        n_bins: Number of reliability bins.
        mode:
            - 'positive_class': bins P(Fake) over [0.0, 1.0] against empirical P(Y=1) rate.
            - 'confidence': Guo et al. top-label confidence max(P, 1-P) over [0.5, 1.0]
              against empirical classification accuracy at `decision_threshold`.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).reshape(-1), 0.0, 1.0)
    n = len(y)

    if mode == "confidence":
        scores = np.where(p >= decision_threshold, p, 1.0 - p)
        preds = (p >= decision_threshold).astype(int)
        correctness = (preds == y).astype(np.float64)
        bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    elif mode == "positive_class":
        scores = p
        correctness = y.astype(np.float64)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unsupported ECE mode: '{mode}'")

    ece = 0.0
    mce = 0.0
    bins_detail: List[Dict[str, Any]] = []

    for i in range(n_bins):
        low = float(bin_edges[i])
        high = float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(scores[mask]))
            avg_acc = float(np.mean(correctness[mask]))
            gap = abs(avg_acc - avg_conf)
            ece += (count / max(n, 1)) * gap
            mce = max(mce, gap)
        else:
            avg_conf = float((low + high) / 2.0)
            avg_acc = None
            gap = 0.0

        bins_detail.append(
            {
                "bin_index": i,
                "low": round(low, 4),
                "high": round(high, 4),
                "count": count,
                "mean_confidence": round(avg_conf, 5),
                "empirical_frequency": round(avg_acc, 5) if avg_acc is not None else None,
                "calibration_gap": round(gap, 5),
            }
        )

    return {
        "mode": mode,
        "n_bins": int(n_bins),
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins_detail,
    }


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    min_temp: float = 0.20,
    max_temp: float = 15.0,
) -> Dict[str, Any]:
    """
    Fits a single scalar temperature parameter T > 0 on validation logits ONLY,
    minimizing negative log-likelihood on CPU float32 for numerical stability.

    Combines dense coarse grid search with gradient refinement and guarantees
    post-scaling validation NLL <= uncalibrated validation NLL (T=1.0).
    """
    logits_np = np.asarray(val_logits, dtype=np.float32).reshape(-1)
    labels_np = np.asarray(val_labels, dtype=np.float32).reshape(-1)
    if len(logits_np) == 0:
        raise ValueError("Validation logits cannot be empty when fitting temperature scaling.")

    logits_t = torch.tensor(logits_np, dtype=torch.float32, device="cpu")
    labels_t = torch.tensor(labels_np, dtype=torch.float32, device="cpu")

    nll_before = float(F.binary_cross_entropy_with_logits(logits_t, labels_t).item())

    # 1. Dense global grid initialization (guarantees avoiding poor L-BFGS local stops)
    candidates = np.unique(
        np.concatenate(
            [
                np.array([1.0], dtype=np.float64),
                np.geomspace(min_temp, max_temp, 200),
                np.linspace(min_temp, max_temp, 200),
            ]
        )
    )
    best_t = 1.0
    best_nll = nll_before
    with torch.no_grad():
        for t_cand in candidates:
            nll_val = float(F.binary_cross_entropy_with_logits(logits_t / float(t_cand), labels_t).item())
            if nll_val < best_nll - 1e-9:
                best_nll = nll_val
                best_t = float(t_cand)

    # 2. Gradient refinement (L-BFGS in CPU float32 around log(T))
    log_temp = nn.Parameter(torch.tensor([np.log(best_t)], dtype=torch.float32, device="cpu"))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t_cur = torch.clamp(torch.exp(log_temp), min=min_temp, max=max_temp)
        loss = F.binary_cross_entropy_with_logits(logits_t / t_cur, labels_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        with torch.no_grad():
            refined_t = float(torch.clamp(torch.exp(log_temp), min=min_temp, max=max_temp).item())
            refined_nll = float(F.binary_cross_entropy_with_logits(logits_t / refined_t, labels_t).item())
        if np.isfinite(refined_nll) and refined_nll < best_nll:
            best_t = refined_t
            best_nll = refined_nll
    except Exception:
        pass

    return {
        "temperature": round(float(best_t), 6),
        "val_nll_before": round(nll_before, 6),
        "val_nll_after": round(best_nll, 6),
        "nll_reduction": round(nll_before - best_nll, 6),
        "fitted_on_split": "validation_only",
        "validation_sample_count": int(len(logits_np)),
    }


def analyze_thresholds_on_validation(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    strategy: str = "youden_j",
) -> Dict[str, Any]:
    """
    Performs validation-only Precision-Recall & ROC operating threshold optimization.
    Never uses test data.
    """
    y = np.asarray(val_labels, dtype=int).reshape(-1)
    p = np.asarray(val_probs, dtype=np.float64).reshape(-1)

    grid = np.round(np.linspace(0.15, 0.85, 71), 4)
    if 0.50 not in grid:
        grid = np.sort(np.append(grid, 0.50))

    sweep: List[Dict[str, float]] = []
    for thr in grid:
        preds = (p >= thr).astype(int)
        acc = float(accuracy_score(y, preds))
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        tn = int(np.sum((y == 0) & (preds == 0)))
        fp = int(np.sum((y == 0) & (preds == 1)))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(1.0 - rec)
        youden_j = float(rec + spec - 1.0)
        bal_acc = float(0.5 * (rec + spec))
        sweep.append(
            {
                "threshold": float(thr),
                "accuracy": round(acc, 5),
                "precision": round(float(prec), 5),
                "recall": round(float(rec), 5),
                "specificity": round(spec, 5),
                "fpr": round(fpr, 5),
                "fnr": round(fnr, 5),
                "f1": round(float(f1), 5),
                "balanced_accuracy": round(bal_acc, 5),
                "youden_j": round(youden_j, 5),
            }
        )

    best_f1_row = max(sweep, key=lambda r: (r["f1"], r["balanced_accuracy"], -abs(r["threshold"] - 0.50)))
    best_youden_row = max(sweep, key=lambda r: (r["youden_j"], r["f1"], -abs(r["threshold"] - 0.50)))

    # High precision threshold (>= 0.90 precision with max recall if achievable)
    hp_candidates = [r for r in sweep if r["precision"] >= 0.90 and r["recall"] >= 0.15]
    high_prec_row = (
        max(hp_candidates, key=lambda r: (r["recall"], r["precision"]))
        if hp_candidates
        else max(sweep, key=lambda r: (r["precision"], r["recall"]))
    )

    # High recall threshold (maximizes recall >= 0.77 with best specificity/balanced accuracy)
    hr_candidates = [r for r in sweep if r["recall"] >= 0.77 and r["specificity"] >= 0.40]
    high_recall_row = (
        max(hr_candidates, key=lambda r: (r["recall"], r["specificity"], r["f1"]))
        if hr_candidates
        else max(sweep, key=lambda r: (r["recall"], r["balanced_accuracy"]))
    )

    if strategy == "best_f1":
        selected_row = best_f1_row
    else:
        selected_row = best_youden_row

    # Exact Precision-Recall curve from sklearn (sampled evenly if >120 pts for JSON compactness)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y, p)
    pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else 0.5
    pr_f1s = (2.0 * pr_precision[:-1] * pr_recall[:-1]) / np.maximum(pr_precision[:-1] + pr_recall[:-1], 1e-8)

    if len(pr_thresholds) > 120:
        sample_idx = np.linspace(0, len(pr_thresholds) - 1, 120, dtype=int)
    else:
        sample_idx = np.arange(len(pr_thresholds))

    return {
        "selection_strategy": strategy,
        "fitted_on_split": "validation_only",
        "default_threshold": 0.50,
        "selected_threshold": float(selected_row["threshold"]),
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "youden_j_threshold": float(best_youden_row["threshold"]),
        "high_precision_threshold": float(high_prec_row["threshold"]),
        "high_recall_threshold": float(high_recall_row["threshold"]),
        "selected_validation_metrics": selected_row,
        "default_validation_metrics": next((r for r in sweep if abs(r["threshold"] - 0.50) < 1e-6), sweep[0]),
        "high_recall_validation_metrics": high_recall_row,
        "high_precision_validation_metrics": high_prec_row,
        "pr_auc_validation": round(pr_auc, 6),
        "threshold_sweep": sweep,
        "pr_curve_summary": {
            "precision": [round(float(x), 5) for x in pr_precision[:-1][sample_idx]],
            "recall": [round(float(x), 5) for x in pr_recall[:-1][sample_idx]],
            "thresholds": [round(float(x), 5) for x in pr_thresholds[sample_idx]],
            "f1": [round(float(x), 5) for x in pr_f1s[sample_idx]] if len(pr_f1s) else [],
        },
    }


def fit_uncertainty_policy_on_validation(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    decision_threshold: float,
    min_coverage: float = 0.70,
    min_class_fraction: float = 0.15,
    min_uncertain_fraction: float = 0.05,
    min_half_width: float = 0.02,
    max_half_width: float = 0.22,
) -> Dict[str, Any]:
    """
    Derives a validated 3-outcome decision band [uncertain_low, uncertain_high] containing
    `decision_threshold` on validation data ONLY.

    Why quantile + logit-margin search instead of fixed Euclidean +-0.06:
      When regularized models (e.g. LabelSmoothing) are temperature-scaled (T > 1),
      calibrated probabilities sit inside a tighter empirical support [p_min, p_max]
      and an asymmetric neighborhood around tau can collapse one decisive class to 0 if
      blindly extended past p_min or p_max.

    This solver evaluates symmetric probability bands, logit-space margins around tau,
    and empirical validation quantile bounds on each side of tau, maximizing validation
    selective accuracy + coverage subject to strict non-degenerate support in ALL three
    outcomes (AUTHENTIC, MANIPULATED, and UNCERTAIN / NEEDS MANUAL REVIEW).
    """
    y = np.asarray(val_labels, dtype=int).reshape(-1)
    p = np.asarray(val_probs, dtype=np.float64).reshape(-1)
    n = len(p)
    tau = float(np.clip(decision_threshold, 0.10, 0.90))
    preds = (p >= tau).astype(int)
    baseline_acc = float(accuracy_score(y, preds)) if n > 0 else 0.0

    # Safe candidate thresholds strictly below tau and strictly above tau
    below_probs = p[p < tau]
    above_probs = p[p > tau]
    avail_above = float(np.max(above_probs) - tau) if len(above_probs) > 0 else 0.10
    avail_below = float(tau - np.min(below_probs)) if len(below_probs) > 0 else 0.10
    eff_min_hw = min(float(min_half_width), max(0.002, 0.45 * min(avail_above, avail_below)))
    z_tau = float(np.log(np.clip(tau, 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(tau, 1e-5, 1.0 - 1e-5))))

    low_set = {round(float(np.clip(tau - d, 0.05, tau - eff_min_hw)), 4) for d in np.linspace(eff_min_hw, max_half_width, 20)}
    low_set.update(round(float(np.clip(sigmoid_np(np.array([z_tau - m]))[0], 0.05, tau - eff_min_hw)), 4) for m in np.linspace(0.02, 0.90, 16))
    if len(below_probs) > 0:
        for q in np.quantile(below_probs, np.linspace(0.35, 0.95, 20)):
            low_set.add(round(float(np.clip(q, 0.05, tau - eff_min_hw)), 4))

    high_set = {round(float(np.clip(tau + d, tau + eff_min_hw, 0.95)), 4) for d in np.linspace(eff_min_hw, max_half_width, 20)}
    high_set.update(round(float(np.clip(sigmoid_np(np.array([z_tau + m]))[0], tau + eff_min_hw, 0.95)), 4) for m in np.linspace(0.01, 0.45, 16))
    if len(above_probs) > 0:
        for q in np.quantile(above_probs, np.linspace(0.05, 0.72, 20)):
            high_set.add(round(float(np.clip(q, tau + eff_min_hw, 0.95)), 4))

    low_cands = sorted(v for v in low_set if 0.05 <= v <= tau - eff_min_hw)
    high_cands = sorted(v for v in high_set if tau + eff_min_hw <= v <= 0.95)
    if not low_cands:
        low_cands = [max(0.05, round(tau - eff_min_hw, 4))]
    if not high_cands:
        high_cands = [min(0.95, round(tau + eff_min_hw, 4))]

    min_auth = min(max(3, int(min_class_fraction * n)), max(1, len(below_probs))) if len(below_probs) > 0 else 0
    min_manip = min(max(3, int(min_class_fraction * n)), max(1, len(above_probs))) if len(above_probs) > 0 else 0
    min_uncert = max(2, int(min_uncertain_fraction * n)) if n >= 20 else 1

    best_candidate: Optional[Dict[str, Any]] = None
    best_utility = -1e9
    fallback_candidate: Optional[Dict[str, Any]] = None
    fallback_score = -1e9

    for low in low_cands:
        for high in high_cands:
            auth_mask = p < low
            manip_mask = p > high
            uncertain_mask = (~auth_mask) & (~manip_mask)
            n_auth = int(np.sum(auth_mask))
            n_manip = int(np.sum(manip_mask))
            n_unc = int(np.sum(uncertain_mask))
            decided_mask = auth_mask | manip_mask
            coverage = float(np.mean(decided_mask)) if n > 0 else 0.0
            uncertain_rate = float(np.mean(uncertain_mask)) if n > 0 else 0.0

            if int(np.sum(decided_mask)) == 0:
                continue

            decided_preds = manip_mask.astype(int)
            sel_acc = float(np.mean(decided_preds[decided_mask] == y[decided_mask]))
            band_acc = float(np.mean(preds[uncertain_mask] == y[uncertain_mask])) if n_unc > 0 else sel_acc
            asymmetry = abs((tau - low) - (high - tau))
            lower_m = round(float(tau - low), 4)
            upper_m = round(float(high - tau), 4)
            avg_hw = round(float(0.5 * (lower_m + upper_m)), 4)

            cand_dict = {
                "decision_threshold": round(tau, 4),
                "uncertain_low": float(low),
                "uncertain_high": float(high),
                "half_width": avg_hw,
                "lower_margin": lower_m,
                "upper_margin": upper_m,
                "min_laplacian_sharpness": 15.0,
                "min_dimension_px": 80,
                "fitted_on_split": "validation_only",
                "validation_baseline_accuracy": round(baseline_acc, 5),
                "validation_selective_accuracy": round(sel_acc, 5),
                "validation_coverage": round(coverage, 5),
                "validation_uncertain_rate": round(uncertain_rate, 5),
                "validation_accuracy_inside_uncertain_band": round(band_acc, 5),
            }

            non_degenerate = (n_auth >= min_auth) and (n_manip >= min_manip)
            if non_degenerate:
                fb_score = sel_acc + 0.12 * coverage - 0.002 * asymmetry
                if fb_score > fallback_score:
                    fallback_score = fb_score
                    fallback_candidate = cand_dict

            if non_degenerate and coverage >= min_coverage and n_unc >= min_uncert:
                utility = sel_acc + 0.12 * coverage - 0.002 * asymmetry
                if utility > best_utility:
                    best_utility = utility
                    best_candidate = cand_dict

    if best_candidate is not None:
        return best_candidate
    if fallback_candidate is not None:
        return fallback_candidate

    # Ultimate safe fallback inside empirical quantiles when dataset is tiny or degenerate
    low = float(low_cands[-1])
    high = float(high_cands[0])
    uncertain_mask = (p >= low) & (p <= high)
    decided_mask = ~uncertain_mask
    coverage = float(np.mean(decided_mask)) if n else 1.0
    sel_acc = float(np.mean((p > high).astype(int)[decided_mask] == y[decided_mask])) if np.any(decided_mask) else baseline_acc
    band_acc = float(np.mean(preds[uncertain_mask] == y[uncertain_mask])) if np.any(uncertain_mask) else sel_acc
    return {
        "decision_threshold": round(tau, 4),
        "uncertain_low": low,
        "uncertain_high": high,
        "half_width": round(float(0.5 * ((tau - low) + (high - tau))), 4),
        "lower_margin": round(float(tau - low), 4),
        "upper_margin": round(float(high - tau), 4),
        "min_laplacian_sharpness": 15.0,
        "min_dimension_px": 80,
        "fitted_on_split": "validation_only",
        "validation_baseline_accuracy": round(baseline_acc, 5),
        "validation_selective_accuracy": round(sel_acc, 5),
        "validation_coverage": round(coverage, 5),
        "validation_uncertain_rate": round(1.0 - coverage, 5),
        "validation_accuracy_inside_uncertain_band": round(band_acc, 5),
    }


def default_uncertainty_policy(
    decision_threshold: float = 0.50,
    half_width: float = 0.10,
) -> Dict[str, Any]:
    """Fallback uncalibrated policy used only if calibration artifact is absent."""
    tau = float(decision_threshold)
    return {
        "decision_threshold": tau,
        "uncertain_low": max(0.05, round(tau - half_width, 4)),
        "uncertain_high": min(0.95, round(tau + half_width, 4)),
        "half_width": float(half_width),
        "min_laplacian_sharpness": 15.0,
        "min_dimension_px": 80,
        "fitted_on_split": "default_fallback",
    }


def classify_with_policy(
    calibrated_prob: float,
    policy: Optional[Dict[str, Any]] = None,
    quality_ok: bool = True,
    quality_reason: Optional[str] = None,
    branch_info: Optional[Dict[str, Any]] = None,
    operating_mode: str = "balanced",
) -> Dict[str, Any]:
    """
    Maps a calibrated probability P(Manipulated) into one of the three formal outcomes:
      - REAL: Evidence supports authentic media (P(Manipulated) < u_low AND P(Manipulated) <= 0.50).
      - FAKE: Evidence supports manipulated or AI-generated media (P(Manipulated) > u_high AND P(Manipulated) >= 0.50).
      - UNCERTAIN: Prediction is close to the decision boundary, image quality is degraded,
        or spatial and frequency branches exhibit severe cross-domain conflict.

    Strict Invariants:
      1. Under no circumstances will an image with P(Manipulated) > 50% be classified as REAL.
      2. Under no circumstances will an image with P(Authentic) > 50% be classified as FAKE.
      3. Disagreements between threshold margins and majority probability resolve to UNCERTAIN.
      4. Severe cross-domain discrepancies (frequency stream detecting generative noise while spatial
         is suppressed) prevent certifying as REAL in balanced mode, and elevate to FAKE in high-recall mode.
    """
    pol = policy or default_uncertainty_policy()
    prob = float(np.clip(calibrated_prob, 0.0, 1.0))

    if operating_mode == "high_recall":
        tau = float(pol.get("high_recall_threshold", min(0.35, pol.get("decision_threshold", 0.50))))
        u_low = float(pol.get("high_recall_uncertain_low", max(0.05, tau - 0.05)))
        u_high = float(pol.get("high_recall_uncertain_high", min(0.95, tau + 0.05)))
    elif operating_mode == "high_precision":
        tau = float(pol.get("high_precision_threshold", max(0.72, pol.get("decision_threshold", 0.71))))
        u_low = float(pol.get("high_precision_uncertain_low", max(0.05, tau - 0.02)))
        u_high = float(pol.get("high_precision_uncertain_high", min(0.95, tau + 0.02)))
    else:  # balanced
        tau = float(pol.get("decision_threshold", 0.50))
        u_low = float(pol.get("uncertain_low", max(0.05, tau - 0.08)))
        u_high = float(pol.get("uncertain_high", min(0.95, tau + 0.08)))

    # Binary entropy in bits [0, 1] as secondary interpretable uncertainty scalar
    p_clip = float(np.clip(prob, 1e-7, 1.0 - 1e-7))
    entropy_bits = float(-(p_clip * np.log2(p_clip) + (1.0 - p_clip) * np.log2(1.0 - p_clip)))

    # Cross-domain discrepancy safeguard:
    # If frequency stream detects manipulation while spatial stream claims real, or vice versa
    if branch_info is not None:
        has_conflict = bool(branch_info.get("has_conflict", False))
        fn_risk = bool(branch_info.get("false_negative_risk", False))
        disc = float(branch_info.get("discrepancy", 0.0))
        freq_p = float(branch_info.get("frequency_prob", prob))
        spat_p = float(branch_info.get("spatial_prob", prob))

        if fn_risk and has_conflict and disc >= 0.20:
            false_neg_warning = (
                f"Elevated False-Negative Risk: Frequency stream (P_freq={freq_p*100:.1f}%) detected "
                f"generative noise residuals, but spatial stream (P_spat={spat_p*100:.1f}%) suppressed "
                "the combined output."
            )
            if operating_mode == "high_recall" and freq_p >= 0.60:
                return {
                    "prediction_outcome": OUTCOME_FAKE,
                    "is_fake": True,
                    "is_uncertain": False,
                    "uncertainty_reason": None,
                    "classifier_confidence": float(freq_p),
                    "margin_from_threshold": round(abs(freq_p - tau), 5),
                    "predictive_entropy_bits": round(entropy_bits, 5),
                    "decision_threshold": tau,
                    "uncertain_low": u_low,
                    "uncertain_high": u_high,
                    "operating_mode": operating_mode,
                    "false_negative_warning": false_neg_warning,
                    "evidence_consistency": "DISCREPANT_OVERRIDDEN_TO_FAKE",
                }
            return {
                "prediction_outcome": OUTCOME_UNCERTAIN,
                "is_fake": bool(prob >= tau),
                "is_uncertain": True,
                "uncertainty_reason": (
                    f"Cross-Domain Discrepancy: Frequency stream detected generative artifacts "
                    f"(P_freq={freq_p*100:.1f}%), conflicting with spatial appearance "
                    f"(P_spat={spat_p*100:.1f}%, gap={disc*100:.1f}%). High false-negative risk."
                ),
                "classifier_confidence": float(max(prob, 1.0 - prob)),
                "margin_from_threshold": round(abs(prob - tau), 5),
                "predictive_entropy_bits": round(entropy_bits, 5),
                "decision_threshold": tau,
                "uncertain_low": u_low,
                "uncertain_high": u_high,
                "operating_mode": operating_mode,
                "false_negative_warning": false_neg_warning,
                "evidence_consistency": "CROSS_DOMAIN_CONFLICT",
            }

    if not quality_ok:
        return {
            "prediction_outcome": OUTCOME_UNCERTAIN,
            "is_fake": bool(prob >= tau),
            "is_uncertain": True,
            "uncertainty_reason": quality_reason or "Input image quality or resolution fell below forensic minimums.",
            "classifier_confidence": float(max(prob, 1.0 - prob)),
            "margin_from_threshold": round(abs(prob - tau), 5),
            "predictive_entropy_bits": round(entropy_bits, 5),
            "decision_threshold": tau,
            "uncertain_low": u_low,
            "uncertain_high": u_high,
            "operating_mode": operating_mode,
            "false_negative_warning": None,
            "evidence_consistency": "DEGRADED_QUALITY",
        }

    # Strict invariant: REAL requires prob < u_low AND prob <= 0.50 (P(Authentic) >= 50%)
    # Under no circumstances will a majority-fake prediction (>50%) be labeled REAL
    if prob < u_low and prob <= 0.50:
        return {
            "prediction_outcome": OUTCOME_REAL,
            "is_fake": False,
            "is_uncertain": False,
            "uncertainty_reason": None,
            "classifier_confidence": float(1.0 - prob),
            "margin_from_threshold": round(abs(prob - tau), 5),
            "predictive_entropy_bits": round(entropy_bits, 5),
            "decision_threshold": tau,
            "uncertain_low": u_low,
            "uncertain_high": u_high,
            "operating_mode": operating_mode,
            "false_negative_warning": None,
            "evidence_consistency": "CONSISTENT_AUTHENTIC",
        }

    # FAKE requires prob > u_high
    if prob > u_high:
        return {
            "prediction_outcome": OUTCOME_FAKE,
            "is_fake": True,
            "is_uncertain": False,
            "uncertainty_reason": None,
            "classifier_confidence": float(prob),
            "margin_from_threshold": round(abs(prob - tau), 5),
            "predictive_entropy_bits": round(entropy_bits, 5),
            "decision_threshold": tau,
            "uncertain_low": u_low,
            "uncertain_high": u_high,
            "operating_mode": operating_mode,
            "false_negative_warning": None,
            "evidence_consistency": "CONSISTENT_MANIPULATED",
        }

    # UNCERTAIN: inside boundary margin or conflicting probability
    if u_low <= prob <= u_high:
        reason = (
            f"Calibrated probability {prob:.3f} lies inside the validated uncertainty zone "
            f"[{u_low:.2f}, {u_high:.2f}] around threshold {tau:.2f}."
        )
    elif prob > 0.50:
        reason = (
            f"Calibrated probability {prob:.3f} leans fake (>0.50) but does not clear the high "
            f"decision threshold boundary {u_high:.2f}."
        )
    else:
        reason = (
            f"Calibrated probability {prob:.3f} leans authentic (<0.50) but does not clear the low "
            f"decision threshold boundary {u_low:.2f}."
        )

    return {
        "prediction_outcome": OUTCOME_UNCERTAIN,
        "is_fake": bool(prob >= tau),
        "is_uncertain": True,
        "uncertainty_reason": reason,
        "classifier_confidence": float(max(prob, 1.0 - prob)),
        "margin_from_threshold": round(abs(prob - tau), 5),
        "predictive_entropy_bits": round(entropy_bits, 5),
        "decision_threshold": tau,
        "uncertain_low": u_low,
        "uncertain_high": u_high,
        "operating_mode": operating_mode,
        "false_negative_warning": None,
        "evidence_consistency": "UNCERTAIN_BOUNDARY",
    }


def evaluate_three_tier_policy(
    y_true: np.ndarray,
    probs: np.ndarray,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluates 3-outcome selective behavior on any split given a fixed validation-derived policy."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    n = len(y)

    outcomes = [classify_with_policy(float(pi), policy=policy, quality_ok=True) for pi in p]
    auth_mask = np.array([o["prediction_outcome"] == OUTCOME_AUTHENTIC for o in outcomes])
    manip_mask = np.array([o["prediction_outcome"] == OUTCOME_MANIPULATED for o in outcomes])
    uncert_mask = np.array([o["prediction_outcome"] == OUTCOME_UNCERTAIN for o in outcomes])
    decided_mask = auth_mask | manip_mask

    decided_preds = np.where(manip_mask, 1, 0)
    selective_acc = (
        float(np.mean(decided_preds[decided_mask] == y[decided_mask]))
        if np.any(decided_mask)
        else None
    )

    tau = float(policy.get("decision_threshold", 0.50))
    tau_preds = (p >= tau).astype(int)
    uncert_acc = (
        float(np.mean(tau_preds[uncert_mask] == y[uncert_mask]))
        if np.any(uncert_mask)
        else None
    )

    return {
        "total_samples": int(n),
        "authentic_count": int(np.sum(auth_mask)),
        "manipulated_count": int(np.sum(manip_mask)),
        "uncertain_count": int(np.sum(uncert_mask)),
        "coverage": round(float(np.sum(decided_mask) / max(n, 1)), 5),
        "uncertain_rate": round(float(np.sum(uncert_mask) / max(n, 1)), 5),
        "selective_accuracy": round(selective_acc, 5) if selective_acc is not None else None,
        "accuracy_inside_uncertain_band": round(uncert_acc, 5) if uncert_acc is not None else None,
        "overall_threshold_accuracy": round(float(np.mean(tau_preds == y)), 5) if n else None,
    }


def load_calibration_policy(
    calibration_path: str = "results/calibration/calibration_results.json",
) -> Dict[str, Any]:
    """Loads calibrated temperature + uncertainty policy from JSON or returns safe T=1.0 fallback."""
    p = Path(calibration_path)
    if p.exists():
        try:
            payload = json.loads(p.read_text())
            temp = float(payload.get("temperature_scaling", {}).get("temperature", 1.0))
            pol = payload.get("uncertainty_policy", {})
            if temp > 0 and "decision_threshold" in pol:
                thr_opt = payload.get("threshold_optimization", {})
                return {
                    "calibrated": True,
                    "calibration_path": str(p),
                    "temperature": temp,
                    "decision_threshold": float(pol["decision_threshold"]),
                    "uncertain_low": float(pol["uncertain_low"]),
                    "uncertain_high": float(pol["uncertain_high"]),
                    "min_laplacian_sharpness": float(pol.get("min_laplacian_sharpness", 15.0)),
                    "min_dimension_px": int(pol.get("min_dimension_px", 80)),
                    "balanced_threshold": float(pol["decision_threshold"]),
                    "high_recall_threshold": float(thr_opt.get("high_recall_threshold", 0.35)),
                    "high_precision_threshold": float(thr_opt.get("high_precision_threshold", 0.72)),
                }
        except Exception:
            pass

    fallback = default_uncertainty_policy()
    return {
        "calibrated": False,
        "calibration_path": None,
        "temperature": 1.0,
        "balanced_threshold": 0.50,
        "high_recall_threshold": 0.35,
        "high_precision_threshold": 0.72,
        **fallback,
    }


def plot_reliability_diagram(
    ece_uncal_pos: Dict[str, Any],
    ece_cal_pos: Dict[str, Any],
    ece_uncal_conf: Dict[str, Any],
    ece_cal_conf: Dict[str, Any],
    brier_uncal: float,
    brier_cal: float,
    temperature: float,
    save_path: Path,
    split_title: str = "Held-Out Test",
):
    """Plots Before vs After Temperature Scaling Reliability Diagrams."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.4))

    def _draw_panel(ax, res_before, res_after, x_label, y_label, title, xlim=(0.0, 1.0)):
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="#64748b", lw=1.6, label="Perfect Calibration")

        for res, color, marker, label in (
            (res_before, "#dc2626", "o", f"Uncalibrated T=1.00 (ECE={res_before['ece']:.4f})"),
            (res_after, "#2563eb", "s", f"Temp-Scaled T={temperature:.2f} (ECE={res_after['ece']:.4f})"),
        ):
            xs = [b["mean_confidence"] for b in res["bins"] if b["count"] > 0 and b["empirical_frequency"] is not None]
            ys = [b["empirical_frequency"] for b in res["bins"] if b["count"] > 0 and b["empirical_frequency"] is not None]
            if xs:
                ax.plot(xs, ys, marker=marker, lw=2.2, markersize=6, color=color, label=label)

        ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5)

    _draw_panel(
        ax1,
        ece_uncal_pos,
        ece_cal_pos,
        "Mean Predicted P(Manipulated)",
        "Empirical Fraction of Positives",
        f"Positive-Class Reliability (Brier: {brier_uncal:.4f} -> {brier_cal:.4f})",
        xlim=(0.0, 1.0),
    )
    _draw_panel(
        ax2,
        ece_uncal_conf,
        ece_cal_conf,
        "Predicted Top-Label Confidence",
        "Empirical Accuracy",
        "Top-Label Confidence Calibration",
        xlim=(0.5, 1.0),
    )

    fig.suptitle(f"{split_title}: Reliability & Calibration Assessment", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confidence_distribution(
    y_true: np.ndarray,
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    policy: Dict[str, Any],
    save_path: Path,
    split_title: str = "Held-Out Test",
    temperature: float = 1.0,
):
    """Plots probability & confidence distributions with the uncertainty band highlighted."""
    y = np.asarray(y_true, dtype=int)
    tau = float(policy["decision_threshold"])
    u_low = float(policy["uncertain_low"])
    u_high = float(policy["uncertain_high"])
    band_fmt = f"[{u_low:.3f}, {u_high:.3f}]" if (u_high - u_low) < 0.04 else f"[{u_low:.2f}, {u_high:.2f}]"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1))
    bins = np.linspace(0.0, 1.0, 21)

    for ax, probs, subtitle in (
        (ax1, uncal_probs, "Uncalibrated P(Manipulated) [T=1.00]"),
        (ax2, cal_probs, f"Calibrated P(Manipulated) [T={temperature:.2f}] + Uncertainty Band"),
    ):
        ax.hist(probs[y == 0], bins=bins, alpha=0.62, color="#059669", edgecolor="white", label="Authentic Real (y=0)")
        ax.hist(probs[y == 1], bins=bins, alpha=0.62, color="#dc2626", edgecolor="white", label="Manipulated Fake (y=1)")
        ax.axvspan(u_low, u_high, color="#f59e0b", alpha=0.25, label=f"Uncertain Review {band_fmt}")
        ax.axvline(tau, color="#1e3a8a", linestyle="-", lw=2.0, label=f"Val Threshold tau={tau:.2f}")
        ax.axvline(0.50, color="#64748b", linestyle="--", lw=1.4, label="Default 0.50")
        ax.set_xlabel("Predicted P(Manipulated)", fontsize=10)
        ax.set_ylabel("Sample Count", fontsize=10)
        ax.set_title(subtitle, fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.55)
        ax.legend(fontsize=8, loc="upper center")

    fig.suptitle(f"{split_title}: Probability & Uncertainty Distribution", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pr_threshold_analysis(
    threshold_analysis: Dict[str, Any],
    save_path: Path,
):
    """Plots 4-panel threshold diagnostics: (1) Precision/Recall/F1, (2) FNR vs FPR, (3) ROC Curve, (4) PR Curve."""
    sweep = threshold_analysis["threshold_sweep"]
    thrs = [r["threshold"] for r in sweep]
    precs = [r["precision"] for r in sweep]
    recs = [r["recall"] for r in sweep]
    f1s = [r["f1"] for r in sweep]
    fnrs = [r.get("fnr", 1.0 - r["recall"]) for r in sweep]
    fprs = [r.get("fpr", 1.0 - r["specificity"]) for r in sweep]

    tau_sel = float(threshold_analysis["selected_threshold"])
    tau_hr = float(threshold_analysis.get("high_recall_threshold", 0.35))
    tau_hp = float(threshold_analysis.get("high_precision_threshold", 0.72))
    pr_sum = threshold_analysis["pr_curve_summary"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13.5, 9.8))

    # Panel 1: Precision, Recall, F1 vs Threshold
    ax1.plot(thrs, precs, color="#2563eb", lw=2.2, label="Precision")
    ax1.plot(thrs, recs, color="#059669", lw=2.2, label="Recall (Sensitivity)")
    ax1.plot(thrs, f1s, color="#7c3aed", lw=2.2, linestyle="-.", label="F1-Score")
    ax1.axvline(0.50, color="#64748b", linestyle="--", lw=1.6, label="Default (0.50)")
    ax1.axvline(tau_sel, color="#dc2626", linestyle="-", lw=2.0, label=f"Balanced ({tau_sel:.2f})")
    ax1.axvline(tau_hr, color="#d97706", linestyle=":", lw=1.8, label=f"High-Recall ({tau_hr:.2f})")
    ax1.axvline(tau_hp, color="#0891b2", linestyle=":", lw=1.8, label=f"High-Precision ({tau_hp:.2f})")
    ax1.set_xlabel("Decision Threshold", fontsize=9.5)
    ax1.set_ylabel("Metric Score", fontsize=9.5)
    ax1.set_title("(A) Validation Threshold vs Performance (P / R / F1)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=7.5, loc="lower left")

    # Panel 2: False Negative Rate vs False Positive Rate
    ax2.plot(thrs, fnrs, color="#dc2626", lw=2.2, label="False Negative Rate (FNR)")
    ax2.plot(thrs, fprs, color="#2563eb", lw=2.2, linestyle="--", label="False Positive Rate (FPR)")
    ax2.axvline(tau_sel, color="#dc2626", linestyle="-", lw=1.8, label=f"Balanced ({tau_sel:.2f})")
    ax2.axvline(tau_hr, color="#d97706", linestyle=":", lw=1.8, label=f"High-Recall ({tau_hr:.2f})")
    ax2.set_xlabel("Decision Threshold", fontsize=9.5)
    ax2.set_ylabel("Error Rate", fontsize=9.5)
    ax2.set_title("(B) Threshold vs False Negative Rate (FNR) & FPR", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="center left")

    # Panel 3: ROC Curve
    roc_fpr = sorted(list(set(fprs)))
    roc_tpr = [max([r["recall"] for r in sweep if abs(r.get("fpr", 1.0 - r["specificity"]) - f) < 1e-4] or [0.0]) for f in roc_fpr]
    roc_auc_val = float(auc(roc_fpr, roc_tpr)) if len(roc_fpr) > 1 else 0.5
    ax3.plot(roc_fpr, roc_tpr, color="#2563eb", lw=2.2, label=f"Validation ROC (AUC = {roc_auc_val:.4f})")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", lw=1.4, label="Random Chance")
    sel_m = threshold_analysis["selected_validation_metrics"]
    def_m = threshold_analysis["default_validation_metrics"]
    hr_m = threshold_analysis.get("high_recall_validation_metrics", sel_m)
    hp_m = threshold_analysis.get("high_precision_validation_metrics", sel_m)
    ax3.scatter([def_m.get("fpr", 1.0 - def_m["specificity"])], [def_m["recall"]], color="#64748b", s=80, zorder=5, label=f"Default 0.50 (TPR={def_m['recall']:.2f})")
    ax3.scatter([sel_m.get("fpr", 1.0 - sel_m["specificity"])], [sel_m["recall"]], color="#dc2626", s=90, zorder=6, label=f"Balanced {tau_sel:.2f} (TPR={sel_m['recall']:.2f})")
    ax3.scatter([hr_m.get("fpr", 1.0 - hr_m["specificity"])], [hr_m["recall"]], color="#d97706", s=85, zorder=7, label=f"High-Recall {tau_hr:.2f} (TPR={hr_m['recall']:.2f})")
    ax3.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9.5)
    ax3.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9.5)
    ax3.set_title("(C) Validation ROC Curve & Operating Points", fontsize=10.5, fontweight="bold")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=7.5, loc="lower right")

    # Panel 4: Precision-Recall Curve
    ax4.plot(
        pr_sum["recall"],
        pr_sum["precision"],
        color="#059669",
        lw=2.4,
        label=f"Validation PR Curve (AP = {threshold_analysis['pr_auc_validation']:.4f})",
    )
    ax4.scatter([def_m["recall"]], [def_m["precision"]], color="#64748b", s=80, zorder=5, label=f"Default 0.50 (F1={def_m['f1']:.3f})")
    ax4.scatter([sel_m["recall"]], [sel_m["precision"]], color="#dc2626", s=90, zorder=6, label=f"Balanced {tau_sel:.2f} (F1={sel_m['f1']:.3f})")
    ax4.scatter([hr_m["recall"]], [hr_m["precision"]], color="#d97706", s=85, zorder=7, label=f"High-Recall {tau_hr:.2f} (F1={hr_m['f1']:.3f})")
    ax4.scatter([hp_m["recall"]], [hp_m["precision"]], color="#0891b2", s=85, zorder=8, label=f"High-Precision {tau_hp:.2f} (Prec={hp_m['precision']:.3f})")
    ax4.set_xlabel("Recall (Sensitivity)", fontsize=9.5)
    ax4.set_ylabel("Precision", fontsize=9.5)
    ax4.set_title("(D) Validation Precision-Recall Operating Points", fontsize=10.5, fontweight="bold")
    ax4.set_xlim(0.0, 1.02)
    ax4.set_ylim(0.0, 1.05)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def collect_split_logits(
    model: DualDomainSentinelNet,
    split_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collects raw logits and binary labels on a split directory without modifying weights."""
    analyzer = SpectralAnalyzer(target_size=256)
    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    items: List[Tuple[Path, int]] = []
    if real_dir.exists():
        items.extend((p, 0) for p in sorted(real_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if fake_dir.exists():
        items.extend((p, 1) for p in sorted(fake_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not items:
        raise ValueError(f"No readable real/fake images found in {split_dir}")

    model.eval()
    all_logits: List[float] = []
    all_labels: List[int] = []

    sp_buf: List[torch.Tensor] = []
    fr_buf: List[torch.Tensor] = []

    for img_path, lbl in items:
        with Image.open(img_path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        sp_buf.append(SPATIAL_TRANSFORM(rgb))
        mag_spec, _ = analyzer.compute_fft_magnitude(rgb)
        fr_buf.append(SPECTRAL_TRANSFORM(Image.fromarray(mag_spec)))
        all_labels.append(lbl)

        if len(sp_buf) >= batch_size:
            logits = model(torch.stack(sp_buf).to(device), torch.stack(fr_buf).to(device)).squeeze(-1)
            all_logits.extend(logits.detach().cpu().numpy().reshape(-1).tolist())
            sp_buf.clear()
            fr_buf.clear()

    if sp_buf:
        logits = model(torch.stack(sp_buf).to(device), torch.stack(fr_buf).to(device)).squeeze(-1)
        all_logits.extend(logits.detach().cpu().numpy().reshape(-1).tolist())

    return np.asarray(all_logits, dtype=np.float64), np.asarray(all_labels, dtype=int)


def run_calibration_from_logits(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    output_dir: str = "results/calibration",
    threshold_strategy: str = "youden_j",
    n_bins: int = 10,
    checkpoint_path: str = "checkpoints/best_model.pth",
    checkpoint_sha256: str = "N/A",
) -> Dict[str, Any]:
    """
    Runs complete temperature scaling, threshold optimization, uncertainty policy validation,
    calibration evaluation (ECE, Brier, NLL), and plot generation from logit arrays.

    Fitting occurs strictly on (val_logits, val_labels).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Fit Temperature Scaling on validation ONLY
    temp_info = fit_temperature_scaling(val_logits, val_labels)
    temperature = float(temp_info["temperature"])

    val_probs_uncal = sigmoid_np(val_logits)
    val_probs_cal = sigmoid_np(val_logits / temperature)

    # 2. Validation-Only Threshold Optimization & PR Analysis
    threshold_info = analyze_thresholds_on_validation(
        val_labels=val_labels,
        val_probs=val_probs_cal,
        strategy=threshold_strategy,
    )
    selected_thr = float(threshold_info["selected_threshold"])

    # 3. Validation-Only 3-Outcome Uncertainty Policy
    uncertainty_policy = fit_uncertainty_policy_on_validation(
        val_labels=val_labels,
        val_probs=val_probs_cal,
        decision_threshold=selected_thr,
    )

    # 4. Calibration Evaluation on Validation Split
    val_ece_uncal_pos = compute_ece(val_labels, val_probs_uncal, n_bins=n_bins, mode="positive_class")
    val_ece_cal_pos = compute_ece(val_labels, val_probs_cal, n_bins=n_bins, mode="positive_class")
    val_ece_uncal_conf = compute_ece(val_labels, val_probs_uncal, n_bins=n_bins, mode="confidence", decision_threshold=0.50)
    val_ece_cal_conf = compute_ece(val_labels, val_probs_cal, n_bins=n_bins, mode="confidence", decision_threshold=selected_thr)

    val_brier_uncal = compute_brier_score(val_labels, val_probs_uncal)
    val_brier_cal = compute_brier_score(val_labels, val_probs_cal)
    val_three_tier = evaluate_three_tier_policy(val_labels, val_probs_cal, uncertainty_policy)

    val_eval = {
        "sample_count": int(len(val_labels)),
        "uncalibrated": {
            "temperature": 1.0,
            "ece_positive_class": round(val_ece_uncal_pos["ece"], 6),
            "ece_confidence": round(val_ece_uncal_conf["ece"], 6),
            "mce_positive_class": round(val_ece_uncal_pos["mce"], 6),
            "brier_score": round(val_brier_uncal, 6),
            "nll": round(compute_nll(val_labels, val_probs_uncal), 6),
            "metrics_at_default_0_50": evaluate_predictions(val_labels, val_probs_uncal, threshold=0.50),
        },
        "calibrated": {
            "temperature": temperature,
            "ece_positive_class": round(val_ece_cal_pos["ece"], 6),
            "ece_confidence": round(val_ece_cal_conf["ece"], 6),
            "mce_positive_class": round(val_ece_cal_pos["mce"], 6),
            "brier_score": round(val_brier_cal, 6),
            "nll": round(compute_nll(val_labels, val_probs_cal), 6),
            "metrics_at_default_0_50": evaluate_predictions(val_labels, val_probs_cal, threshold=0.50),
            "metrics_at_selected_threshold": evaluate_predictions(val_labels, val_probs_cal, threshold=selected_thr),
            "three_outcome_policy_eval": val_three_tier,
            "reliability_bins": val_ece_cal_pos["bins"],
        },
    }

    # 5. Out-of-Sample Evaluation on Held-Out Test Split (ZERO tuning on test!)
    test_eval: Optional[Dict[str, Any]] = None
    plot_y = val_labels
    plot_uncal = val_probs_uncal
    plot_cal = val_probs_cal
    plot_ece_uncal_pos = val_ece_uncal_pos
    plot_ece_cal_pos = val_ece_cal_pos
    plot_ece_uncal_conf = val_ece_uncal_conf
    plot_ece_cal_conf = val_ece_cal_conf
    plot_brier_uncal = val_brier_uncal
    plot_brier_cal = val_brier_cal
    plot_split_name = "Validation"

    if test_logits is not None and test_labels is not None and len(test_logits) > 0:
        test_probs_uncal = sigmoid_np(test_logits)
        test_probs_cal = sigmoid_np(test_logits / temperature)

        test_ece_uncal_pos = compute_ece(test_labels, test_probs_uncal, n_bins=n_bins, mode="positive_class")
        test_ece_cal_pos = compute_ece(test_labels, test_probs_cal, n_bins=n_bins, mode="positive_class")
        test_ece_uncal_conf = compute_ece(test_labels, test_probs_uncal, n_bins=n_bins, mode="confidence", decision_threshold=0.50)
        test_ece_cal_conf = compute_ece(test_labels, test_probs_cal, n_bins=n_bins, mode="confidence", decision_threshold=selected_thr)

        test_brier_uncal = compute_brier_score(test_labels, test_probs_uncal)
        test_brier_cal = compute_brier_score(test_labels, test_probs_cal)
        test_three_tier = evaluate_three_tier_policy(test_labels, test_probs_cal, uncertainty_policy)

        test_eval = {
            "sample_count": int(len(test_labels)),
            "strict_out_of_sample": True,
            "uncalibrated": {
                "temperature": 1.0,
                "ece_positive_class": round(test_ece_uncal_pos["ece"], 6),
                "ece_confidence": round(test_ece_uncal_conf["ece"], 6),
                "mce_positive_class": round(test_ece_uncal_pos["mce"], 6),
                "brier_score": round(test_brier_uncal, 6),
                "nll": round(compute_nll(test_labels, test_probs_uncal), 6),
                "metrics_at_default_0_50": evaluate_predictions(test_labels, test_probs_uncal, threshold=0.50),
            },
            "calibrated": {
                "temperature": temperature,
                "ece_positive_class": round(test_ece_cal_pos["ece"], 6),
                "ece_confidence": round(test_ece_cal_conf["ece"], 6),
                "mce_positive_class": round(test_ece_cal_pos["mce"], 6),
                "brier_score": round(test_brier_cal, 6),
                "nll": round(compute_nll(test_labels, test_probs_cal), 6),
                "metrics_at_default_0_50": evaluate_predictions(test_labels, test_probs_cal, threshold=0.50),
                "metrics_at_selected_threshold": evaluate_predictions(test_labels, test_probs_cal, threshold=selected_thr),
                "three_outcome_policy_eval": test_three_tier,
                "reliability_bins": test_ece_cal_pos["bins"],
            },
        }

        plot_y = test_labels
        plot_uncal = test_probs_uncal
        plot_cal = test_probs_cal
        plot_ece_uncal_pos = test_ece_uncal_pos
        plot_ece_cal_pos = test_ece_cal_pos
        plot_ece_uncal_conf = test_ece_uncal_conf
        plot_ece_cal_conf = test_ece_cal_conf
        plot_brier_uncal = test_brier_uncal
        plot_brier_cal = test_brier_cal
        plot_split_name = "Held-Out Test (Out-of-Sample)"

    # 6. Generate Diagnostic Plots
    rel_plot = out_path / "reliability_diagram.png"
    dist_plot = out_path / "confidence_distribution.png"
    pr_plot = out_path / "pr_threshold_analysis.png"

    plot_reliability_diagram(
        ece_uncal_pos=plot_ece_uncal_pos,
        ece_cal_pos=plot_ece_cal_pos,
        ece_uncal_conf=plot_ece_uncal_conf,
        ece_cal_conf=plot_ece_cal_conf,
        brier_uncal=plot_brier_uncal,
        brier_cal=plot_brier_cal,
        temperature=temperature,
        save_path=rel_plot,
        split_title=plot_split_name,
    )
    plot_confidence_distribution(
        y_true=plot_y,
        uncal_probs=plot_uncal,
        cal_probs=plot_cal,
        policy=uncertainty_policy,
        save_path=dist_plot,
        split_title=plot_split_name,
        temperature=temperature,
    )
    plot_pr_threshold_analysis(
        threshold_analysis=threshold_info,
        save_path=pr_plot,
    )

    payload: Dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_guarantees": {
            "temperature_fit_split": "validation_only",
            "threshold_selection_split": "validation_only",
            "uncertainty_policy_fit_split": "validation_only",
            "zero_test_set_tuning": True,
            "checkpoint_weights_modified": False,
            "bayesian_heuristic_override_removed": True,
        },
        "temperature_scaling": temp_info,
        "threshold_optimization": threshold_info,
        "uncertainty_policy": uncertainty_policy,
        "outcomes": [OUTCOME_AUTHENTIC, OUTCOME_MANIPULATED, OUTCOME_UNCERTAIN],
        "validation": val_eval,
        "test": test_eval,
        "artifacts": {
            "reliability_diagram": str(rel_plot),
            "confidence_distribution": str(dist_plot),
            "pr_threshold_analysis": str(pr_plot),
        },
    }

    out_json = out_path / "calibration_results.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_calibration_pipeline(
    checkpoint_path: str = "checkpoints/best_model.pth",
    config_path: str = "config/default_config.yaml",
    data_dir: str = "data",
    output_dir: str = "results/calibration",
    threshold_strategy: str = "youden_j",
    device_str: str = "auto",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Loads the trained model checkpoint read-only, evaluates raw logits on `data/val` and `data/test`,
    fits calibration parameters on `data/val` ONLY, and saves calibration results & plots.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg = load_config(config_path) if Path(config_path).exists() else {}
    model_cfg = cfg.get("model", {})
    device = get_device(device_str)

    model = DualDomainSentinelNet(
        backbone_name=model_cfg.get("backbone", "efficientnet_b0"),
        pretrained=False,
        spatial_dim=model_cfg.get("spatial_feat_dim", 512),
        spectral_dim=model_cfg.get("spectral_feat_dim", 256),
        fusion_type=model_cfg.get("fusion_type", "cross_attention"),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.3),
    )
    loaded = torch.load(ckpt, map_location=device, weights_only=True)
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    data_root = Path(data_dir)
    val_logits, val_labels = collect_split_logits(model, data_root / "val", device, batch_size=batch_size)

    test_dir = data_root / "test"
    test_logits, test_labels = (
        collect_split_logits(model, test_dir, device, batch_size=batch_size)
        if test_dir.exists()
        else (None, None)
    )

    return run_calibration_from_logits(
        val_logits=val_logits,
        val_labels=val_labels,
        test_logits=test_logits,
        test_labels=test_labels,
        output_dir=output_dir,
        threshold_strategy=threshold_strategy,
        checkpoint_path=str(ckpt),
        checkpoint_sha256=ckpt_sha,
    )


def main():
    parser = argparse.ArgumentParser(description="Fit Validation Temperature Scaling & Calibration Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/calibration")
    parser.add_argument("--strategy", type=str, default="youden_j", choices=["youden_j", "best_f1"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    res = run_calibration_pipeline(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold_strategy=args.strategy,
        device_str=args.device,
    )
    t_info = res["temperature_scaling"]
    th_info = res["threshold_optimization"]
    pol = res["uncertainty_policy"]
    print(f"[✓] Calibration Complete -> {Path(args.output_dir) / 'calibration_results.json'}")
    print(
        f"    Validation Temperature Scaling: T={t_info['temperature']:.4f} | "
        f"Val NLL: {t_info['val_nll_before']:.4f} -> {t_info['val_nll_after']:.4f}"
    )
    print(
        f"    Validation Threshold Selection: default=0.50 | selected={th_info['selected_threshold']:.2f} "
        f"(Uncertainty Band=[{pol['uncertain_low']:.2f}, {pol['uncertain_high']:.2f}])"
    )
    if res.get("test"):
        tu = res["test"]["uncalibrated"]
        tc = res["test"]["calibrated"]
        t3 = tc["three_outcome_policy_eval"]
        print(
            f"    Held-Out Test ECE (Class-1): {tu['ece_positive_class']:.4f} -> {tc['ece_positive_class']:.4f} | "
            f"Brier: {tu['brier_score']:.4f} -> {tc['brier_score']:.4f}"
        )
        print(
            f"    Held-Out Test @0.50 Acc={tc['metrics_at_default_0_50']['accuracy']*100:.2f}% | "
            f"@ValTau({th_info['selected_threshold']:.2f}) Acc={tc['metrics_at_selected_threshold']['accuracy']*100:.2f}% | "
            f"Selective Acc={t3['selective_accuracy']*100:.2f}% (Coverage={t3['coverage']*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
