"""
Dedicated False-Negative Forensic Audit Pipeline.

Audits all manipulated/AI-generated images that were misclassified as REAL, isolates
failure modes across spatial, frequency, and cross-attention fusion streams, extracts
visual diagnostic artifacts (Grad-CAM, FFT, ELA, SRM), and produces a structured
forensic breakdown report.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.core.utils import SPATIAL_TRANSFORM, SPECTRAL_TRANSFORM, get_device
from src.evaluation.calibration import load_calibration_policy
from src.pipeline import ForensicInferencePipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_false_negative_audit(
    checkpoint_path: str = "checkpoints/best_model.pth",
    calibration_path: Optional[str] = "results/calibration/calibration_results.json",
    data_dir: str = "data/test/fake",
    additional_images: Optional[List[str]] = None,
    output_dir: str = "results/false_negatives",
    max_cases_to_save: int = 25,
) -> Dict[str, Any]:
    """
    Executes a forensic false-negative audit on manipulated images.
    
    Identifies every sample where Ground Truth = FAKE but Predicted Verdict = REAL,
    saves multi-domain visualizations, and categorizes failure mechanisms.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cases_dir = out_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Initializing Forensic Inference Pipeline with {checkpoint_path}...")
    pipeline = ForensicInferencePipeline(
        checkpoint_path=checkpoint_path,
        calibration_path=calibration_path,
    )

    image_paths: List[Path] = []
    data_path = Path(data_dir)
    if data_path.exists():
        if data_path.is_dir():
            image_paths.extend([p for p in data_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        else:
            image_paths.append(data_path)

    if additional_images:
        for extra in additional_images:
            ep = Path(extra)
            if ep.exists() and ep.suffix.lower() in IMAGE_EXTS:
                image_paths.append(ep)

    print(f"[i] Auditing {len(image_paths)} ground-truth manipulated images for false negatives...")
    t_start = time.time()

    total_evaluated = 0
    fn_under_default_50 = 0
    fn_under_val_tau_71 = 0
    fn_under_balanced_selective = 0
    saved_fn_records: List[Dict[str, Any]] = []

    pattern_counts = {
        "spatial_suppression_conflict": 0,  # Frequency detected fake, but spatial overruled
        "sub_threshold_near_boundary": 0,   # P between 0.50 and 0.71
        "severe_miss": 0,                   # P < 0.40 and both branches low
        "compression_or_blur_degraded": 0,  # Low sharpness or high compression
    }

    for idx, img_p in enumerate(image_paths):
        try:
            with Image.open(img_p) as pil_raw:
                img_rgb = pil_raw.convert("RGB")
                img_copy = img_rgb.copy()
        except Exception as e:
            print(f"[!] Failed to read {img_p}: {e}")
            continue

        total_evaluated += 1
        res = pipeline.analyze_image(img_copy, operating_mode="balanced")

        p_fake = float(res["fake_probability"])
        s_prob = float(res.get("spatial_fake_prob", p_fake))
        f_prob = float(res.get("frequency_fake_prob", p_fake))
        disc = float(res.get("branch_discrepancy", abs(s_prob - f_prob)))
        has_conflict = bool(res.get("has_branch_conflict", False))
        fn_risk = bool(res.get("false_negative_risk", False))
        outcome = res["prediction_outcome"]
        tau = float(res["calibration_metadata"]["decision_threshold"])

        # Ground truth is FAKE (1).
        # False negative under default 0.50 threshold:
        is_fn_50 = p_fake < 0.50
        # False negative under validation-derived 0.71 threshold (binary):
        is_fn_71 = p_fake < tau
        # False negative under 3-tier selective policy (declared REAL):
        is_fn_selective = outcome == "REAL"

        if is_fn_50:
            fn_under_default_50 += 1
        if is_fn_71:
            fn_under_val_tau_71 += 1
        if is_fn_selective:
            fn_under_balanced_selective += 1

        # Determine failure pattern
        if fn_risk and has_conflict and disc >= 0.15:
            pat = "spatial_suppression_conflict"
            pattern_counts["spatial_suppression_conflict"] += 1
        elif 0.50 <= p_fake < tau:
            pat = "sub_threshold_near_boundary"
            pattern_counts["sub_threshold_near_boundary"] += 1
        elif res.get("laplacian_variance", 100.0) < 30.0:
            pat = "compression_or_blur_degraded"
            pattern_counts["compression_or_blur_degraded"] += 1
        else:
            pat = "severe_miss"
            pattern_counts["severe_miss"] += 1

        # If it's a false negative under tau or selective, save details
        if (is_fn_71 or is_fn_selective) and len(saved_fn_records) < max_cases_to_save:
            case_id = f"fn_case_{len(saved_fn_records) + 1:03d}_{img_p.stem[:24]}"
            case_folder = cases_dir / case_id
            case_folder.mkdir(parents=True, exist_ok=True)

            orig_save = case_folder / "original.jpg"
            img_copy.save(orig_save, "JPEG", quality=95)

            cam_save = case_folder / "gradcam_overlay.png"
            res["gradcam_overlay"].save(cam_save)

            fft_save = case_folder / "fft_colormap.png"
            res["fft_log_colormap_img"].save(fft_save)

            ela_save = case_folder / "ela_heatmap.png"
            res["ela_heatmap_color"].save(ela_save)

            srm_save = None
            if res.get("srm_heatmap") is not None:
                srm_save = case_folder / "srm_heatmap.png"
                res["srm_heatmap"].save(srm_save)

            fn_rec = {
                "case_id": case_id,
                "file_path": str(img_p),
                "ground_truth": "FAKE",
                "predicted_outcome": outcome,
                "predicted_label": res["prediction_label"],
                "fake_probability": round(p_fake, 5),
                "uncalibrated_probability": round(float(res["forensic_evidence"]["neural_uncalibrated_prob"]), 5),
                "decision_threshold": round(tau, 3),
                "spatial_branch_fake_prob": round(s_prob, 5),
                "frequency_branch_fake_prob": round(f_prob, 5),
                "branch_discrepancy": round(disc, 5),
                "has_branch_conflict": has_conflict,
                "false_negative_risk": fn_risk,
                "laplacian_sharpness": round(float(res.get("laplacian_variance", 0.0)), 2),
                "failure_pattern": pat,
                "false_negative_warning": res.get("false_negative_warning"),
                "uncertainty_reason": res.get("uncertainty_reason"),
                "artifacts": {
                    "original": str(orig_save.relative_to(out_path.parent.parent)),
                    "gradcam": str(cam_save.relative_to(out_path.parent.parent)),
                    "fft": str(fft_save.relative_to(out_path.parent.parent)),
                    "ela": str(ela_save.relative_to(out_path.parent.parent)),
                    "srm": str(srm_save.relative_to(out_path.parent.parent)) if srm_save else None,
                },
            }
            saved_fn_records.append(fn_rec)

    total_time = time.time() - t_start
    print(f"[✓] Evaluated {total_evaluated} fake images in {total_time:.1f}s.")
    print(f"    FN @ default threshold 0.50: {fn_under_default_50}/{total_evaluated} ({fn_under_default_50/max(total_evaluated,1)*100:.1f}%)")
    print(f"    FN @ val threshold 0.71:    {fn_under_val_tau_71}/{total_evaluated} ({fn_under_val_tau_71/max(total_evaluated,1)*100:.1f}%)")
    print(f"    FN @ selective REAL call:   {fn_under_balanced_selective}/{total_evaluated} ({fn_under_balanced_selective/max(total_evaluated,1)*100:.1f}%)")
    print(f"    Saved detailed case studies: {len(saved_fn_records)} cases in {cases_dir}")

    # Generate diagnostic plot for False-Negative analysis
    plot_fn_diagnostic(
        pattern_counts=pattern_counts,
        fn_50=fn_under_default_50,
        fn_71=fn_under_val_tau_71,
        fn_selective=fn_under_balanced_selective,
        total=total_evaluated,
        output_path=out_path / "false_negative_analysis.png",
    )

    summary_report = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_path": checkpoint_path,
        "total_manipulated_images_evaluated": total_evaluated,
        "false_negatives": {
            "binary_threshold_0_50": {
                "count": fn_under_default_50,
                "rate": round(fn_under_default_50 / max(total_evaluated, 1), 4),
            },
            "binary_threshold_0_71": {
                "count": fn_under_val_tau_71,
                "rate": round(fn_under_val_tau_71 / max(total_evaluated, 1), 4),
            },
            "selective_policy_real_calls": {
                "count": fn_under_balanced_selective,
                "rate": round(fn_under_balanced_selective / max(total_evaluated, 1), 4),
                "improvement_vs_binary_71": f"{(fn_under_val_tau_71 - fn_under_balanced_selective) / max(fn_under_val_tau_71, 1) * 100:.1f}% reduction in false certifications",
            },
        },
        "failure_pattern_distribution": pattern_counts,
        "saved_case_studies_count": len(saved_fn_records),
        "saved_case_studies": saved_fn_records,
    }

    report_json = out_path / "false_negative_report.json"
    with open(report_json, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[✓] Saved False Negative Report -> {report_json}")

    return summary_report


def plot_fn_diagnostic(
    pattern_counts: Dict[str, int],
    fn_50: int,
    fn_71: int,
    fn_selective: int,
    total: int,
    output_path: Path,
) -> None:
    """Generates a 2-panel false-negative diagnostic figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)

    # Panel 1: False Negative Count Comparison across decision policies
    policies = ["Binary τ=0.50", "Binary τ=0.71", "3-Tier Selective"]
    counts = [fn_50, fn_71, fn_selective]
    colors = ["#f59e0b", "#ef4444", "#10b981"]

    bars = axes[0].bar(policies, counts, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[0].set_title("False Negative Comparison across Policies\n(Lower is better)", fontsize=11, fontweight="bold", pad=10)
    axes[0].set_ylabel("False Negatives (Count)", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_ylim(0, max(counts) * 1.25 if counts else 10)

    for bar in bars:
        h = bar.get_height()
        pct = (h / max(total, 1)) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            f"{h} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel 2: Distribution of Failure Patterns
    pat_labels = [
        "Spatial Suppression\n(Frequency FAKE)",
        "Sub-Threshold\n[0.50 - 0.71]",
        "Low Sharpness\n/ Blur",
        "Severe\nMiss",
    ]
    pat_vals = [
        pattern_counts.get("spatial_suppression_conflict", 0),
        pattern_counts.get("sub_threshold_near_boundary", 0),
        pattern_counts.get("compression_or_blur_degraded", 0),
        pattern_counts.get("severe_miss", 0),
    ]
    p_colors = ["#8b5cf6", "#3b82f6", "#06b6d4", "#ec4899"]

    p_bars = axes[1].bar(pat_labels, pat_vals, color=p_colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[1].set_title("Empirical Failure Pattern Distribution\n(Root Cause Breakdown)", fontsize=11, fontweight="bold", pad=10)
    axes[1].set_ylabel("Occurrences", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].set_ylim(0, max(pat_vals) * 1.25 if max(pat_vals) > 0 else 10)

    for bar in p_bars:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.5,
            f"{h}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved False Negative Diagnostic Plot -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedicated False-Negative Forensic Audit")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--calibration", default="results/calibration/calibration_results.json")
    parser.add_argument("--data-dir", default="data/test/fake")
    parser.add_argument("--include-user-images", action="store_true")
    parser.add_argument("--output-dir", default="results/false_negatives")
    parser.add_argument("--max-cases", type=int, default=20)
    args = parser.parse_args()

    extras = []
    if args.include_user_images:
        user_img = "/Users/antrikshmishra/.gemini/antigravity/brain/0bce4474-8b0d-4abc-b2c2-1900bc4214a5/.user_uploaded/media_1794894189108_a38ad3db.jpg"
        if os.path.exists(user_img):
            extras.append(user_img)

    run_false_negative_audit(
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        data_dir=args.data_dir,
        additional_images=extras,
        output_dir=args.output_dir,
        max_cases_to_save=args.max_cases,
    )
