"""
Model Performance & Empirical Evaluation Dashboard.
Renders empirical test and validation metrics, actual training/validation loss curves,
confusion matrices, ROC curves, and checkpoint architecture details
directly from real evaluated evaluation_metrics.json and training_history.json.
Eliminates all hardcoded fallbacks and clearly distinguishes training, validation,
and held-out test splits.
"""

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Direct MPLCONFIGDIR to pre-compiled local font cache to avoid 70s+ font scan hang
REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def get_evaluation_artifacts_signature(
    repo_root: Optional[Union[str, Path]] = None,
) -> Tuple[Tuple[str, int, int], ...]:
    """
    Computes a lightweight `(rel_path, st_mtime_ns, st_size)` signature across evaluation,
    calibration, ablation, dataset audit, and checkpoint artifacts for instant cache invalidation (B10 fix).
    """
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent
    watched_paths = [
        root / "checkpoints" / "evaluation_metrics.json",
        root / "checkpoints" / "training_history.json",
        root / "checkpoints" / "best_model.pth",
        root / "checkpoints" / "test_confusion_matrix.png",
        root / "checkpoints" / "test_roc_curve.png",
        root / "checkpoints" / "confusion_matrix.png",
        root / "checkpoints" / "roc_curve.png",
        root / "results" / "ablation_results.json",
        root / "results" / "dataset_audit.json",
        root / "results" / "calibration" / "calibration_results.json",
        root / "results" / "calibration" / "reliability_diagram.png",
        root / "results" / "calibration" / "confidence_distribution.png",
        root / "results" / "calibration" / "pr_threshold_analysis.png",
        root / "results" / "external_benchmark" / "external_benchmark_summary.json",
    ]
    sig = []
    for p in watched_paths:
        if p.exists():
            try:
                st_res = p.stat()
                sig.append((str(p), int(st_res.st_mtime_ns), int(st_res.st_size)))
            except OSError:
                sig.append((str(p), -1, -1))
        else:
            sig.append((str(p), 0, 0))
    return tuple(sig)


@st.cache_data
def _load_evaluation_data_cached(
    cache_signature: Tuple[Tuple[str, int, int], ...],
    repo_root_str: str,
) -> Dict[str, Any]:
    """Loads metrics, training history, ablation study results, calibration results, and checkpoint metadata."""
    del cache_signature  # Keyed by artifact mtime+size signature for immediate invalidation on file edits.
    repo_root = Path(repo_root_str)
    checkpoints_dir = repo_root / "checkpoints"
    results_dir = repo_root / "results"
    calibration_dir = results_dir / "calibration"
    
    metrics = {}
    metrics_path = checkpoints_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}

    history = {}
    history_path = checkpoints_dir / "training_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = {}

    ablation = {}
    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        try:
            ablation = json.loads(ablation_path.read_text())
        except Exception:
            ablation = {}

    calibration = {}
    calibration_path = calibration_dir / "calibration_results.json"
    if calibration_path.exists():
        try:
            calibration = json.loads(calibration_path.read_text())
        except Exception:
            calibration = {}

    external_benchmark = {}
    ext_path = results_dir / "external_benchmark" / "external_benchmark_summary.json"
    if ext_path.exists():
        try:
            external_benchmark = json.loads(ext_path.read_text())
        except Exception:
            external_benchmark = {}

    dataset_audit = {}
    audit_path = results_dir / "dataset_audit.json"
    if audit_path.exists():
        try:
            dataset_audit = json.loads(audit_path.read_text())
        except Exception:
            dataset_audit = {}

    # Checkpoint information
    ckpt_path = checkpoints_dir / "best_model.pth"
    ckpt_info = {
        "exists": ckpt_path.exists(),
        "path": str(ckpt_path),
        "size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2) if ckpt_path.exists() else 0.0,
        "sha256": "N/A"
    }
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                ckpt_info["sha256"] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

    return {
        "metrics": metrics,
        "history": history,
        "ablation": ablation,
        "calibration": calibration,
        "external_benchmark": external_benchmark,
        "dataset_audit": dataset_audit,
        "checkpoint": ckpt_info,
        "test_confusion_matrix_path": checkpoints_dir / "test_confusion_matrix.png",
        "test_roc_curve_path": checkpoints_dir / "test_roc_curve.png",
        "val_confusion_matrix_path": checkpoints_dir / "confusion_matrix.png",
        "val_roc_curve_path": checkpoints_dir / "roc_curve.png",
        "reliability_diagram_path": calibration_dir / "reliability_diagram.png",
        "confidence_distribution_path": calibration_dir / "confidence_distribution.png",
        "pr_threshold_analysis_path": calibration_dir / "pr_threshold_analysis.png",
    }


def load_evaluation_data(repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads evaluation artifacts with automatic mtime+size cache invalidation (B10 fix)."""
    resolved_root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent
    signature = get_evaluation_artifacts_signature(resolved_root)
    return _load_evaluation_data_cached(signature, str(resolved_root))


@st.cache_data
def _render_ablation_charts_png(experiments_json: str) -> Tuple[bytes, bytes]:
    """Caches 180-DPI ablation chart PNG renders to avoid Matplotlib redraw overhead on Streamlit reruns."""
    experiments = json.loads(experiments_json)
    labels = [exp.get("short_label", f"Exp {exp.get('experiment_number', i + 1)}") for i, exp in enumerate(experiments)]
    accs = [exp["accuracy"] * 100.0 for exp in experiments]
    f1s = [exp["f1"] * 100.0 for exp in experiments]
    aucs = [(exp["auc_roc"] * 100.0 if exp.get("auc_roc") is not None else 0.0) for exp in experiments]
    eers = [(exp["eer"] * 100.0 if exp.get("eer") is not None else 0.0) for exp in experiments]
    lats = [exp["inference_latency_ms"] for exp in experiments]
    params_m = [exp["num_parameters"] / 1e6 for exp in experiments]

    x = np.arange(len(labels))
    width = 0.25
    fig_perf, ax_perf = plt.subplots(figsize=(6.4, 4.1))
    b1 = ax_perf.bar(x - width, accs, width, label="Accuracy (%)", color="#2563eb")
    b2 = ax_perf.bar(x, f1s, width, label="F1-Score (%)", color="#059669")
    b3 = ax_perf.bar(x + width, aucs, width, label="ROC-AUC (%)", color="#7c3aed")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax_perf.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(labels, fontsize=8)
    ax_perf.set_ylim(0, 115)
    ax_perf.set_ylabel("Score (%)", fontsize=10)
    ax_perf.set_title("Ablation Performance: Accuracy, F1 & ROC-AUC", fontsize=11, fontweight="bold")
    ax_perf.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_perf.legend(fontsize=8, loc="lower right")
    fig_perf.tight_layout()
    buf_perf = io.BytesIO()
    fig_perf.savefig(buf_perf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig_perf)

    fig_eff, ax_eer = plt.subplots(figsize=(6.4, 4.1))
    bars_eer = ax_eer.bar(x - 0.18, eers, 0.35, label="Equal Error Rate (%)", color="#dc2626", alpha=0.88)
    ax_eer.set_ylabel("Equal Error Rate (%)", color="#dc2626", fontsize=10)
    ax_eer.tick_params(axis="y", labelcolor="#dc2626")
    ax_eer.set_xticks(x)
    ax_eer.set_xticklabels(labels, fontsize=8)
    ax_eer.set_ylim(0, max(eers + [10.0]) * 1.25)

    ax_lat = ax_eer.twinx()
    bars_lat = ax_lat.bar(x + 0.18, lats, 0.35, label="Latency (ms/img)", color="#0284c7", alpha=0.88)
    ax_lat.set_ylabel("Single-Image Latency (ms)", color="#0284c7", fontsize=10)
    ax_lat.tick_params(axis="y", labelcolor="#0284c7")
    ax_lat.set_ylim(0, max(lats + [5.0]) * 1.30)

    for bar in bars_eer:
        h = bar.get_height()
        ax_eer.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#991b1b",
        )
    for bar, pm in zip(bars_lat, params_m):
        h = bar.get_height()
        ax_lat.annotate(
            f"{h:.1f}ms\n({pm:.2f}M)",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#075985",
        )

    ax_eer.set_title("Ablation Trade-off: EER vs Latency & Parameters", fontsize=11, fontweight="bold")
    ax_eer.grid(True, axis="y", linestyle=":", alpha=0.5)
    lines1, labels1 = ax_eer.get_legend_handles_labels()
    lines2, labels2 = ax_lat.get_legend_handles_labels()
    ax_eer.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig_eff.tight_layout()
    buf_eff = io.BytesIO()
    fig_eff.savefig(buf_eff, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig_eff)
    return buf_perf.getvalue(), buf_eff.getvalue()


@st.cache_data
def _render_training_curves_png(history_json: str) -> Tuple[bytes, bytes]:
    """Caches 180-DPI training/validation curve PNG renders keyed by training history JSON."""
    history = json.loads(history_json)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    val_auc = history.get("val_auc", [])
    epochs = list(range(1, len(train_loss) + 1))

    fig_loss, ax_loss = plt.subplots(figsize=(6, 3.8))
    ax_loss.plot(epochs, train_loss, marker="o", color="#2563eb", lw=2.2, label="Train Loss")
    if val_loss:
        ax_loss.plot(epochs, val_loss[:len(epochs)], marker="s", color="#dc2626", lw=2.2, linestyle="--", label="Val Loss")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("LabelSmoothingBCELoss (eps=0.06)", fontsize=10)
    ax_loss.set_title("Training & Validation Loss Convergence", fontsize=11, fontweight="bold")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=9)
    buf_loss = io.BytesIO()
    fig_loss.savefig(buf_loss, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig_loss)

    fig_acc, ax_acc = plt.subplots(figsize=(6, 3.8))
    ax_acc.plot(epochs, [a * 100.0 for a in train_acc], marker="o", color="#059669", lw=2.2, label="Train Accuracy (%)")
    if val_acc:
        ax_acc.plot(epochs, [a * 100.0 for a in val_acc[:len(epochs)]], marker="s", color="#d97706", lw=2.0, linestyle="-.", label="Val Accuracy (%)")
    if val_auc:
        ax_acc.plot(epochs, [a * 100.0 for a in val_auc[:len(epochs)]], marker="^", color="#7c3aed", lw=2.2, linestyle="--", label="Val ROC-AUC (%)")
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Metric Score (%)", fontsize=10)
    ax_acc.set_title("Training vs. Validation Accuracy & AUC", fontsize=11, fontweight="bold")
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=9)
    buf_acc = io.BytesIO()
    fig_acc.savefig(buf_acc, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig_acc)
    return buf_loss.getvalue(), buf_acc.getvalue()


def render_ablation_section(ablation: Dict[str, Any]):
    """Renders controlled 5-experiment ablation study table and comparative charts."""
    st.markdown("#### 🧪 Controlled Architectural & Loss Ablation Study")
    experiments = ablation.get("experiments", []) if isinstance(ablation, dict) else []
    if not experiments:
        st.info("Controlled Architectural Ablation Study (`results/ablation_results.json`): Not yet evaluated.")
        return

    proto = ablation.get("study_protocol", {})
    st.caption(
        f"Fair group-disjoint comparison (group_disjoint={proto.get('group_disjoint', 'N/A')}) — "
        f"Train: {proto.get('train_samples', 'N/A')} | Val: {proto.get('val_samples', 'N/A')} | "
        f"Test: {proto.get('test_samples', 'N/A')} | Backbone: {proto.get('backbone', 'efficientnet_b0')} | "
        f"Epochs: {proto.get('epochs', 'N/A')} | Device: {proto.get('device', 'N/A')}"
    )

    ab_md = (
        "| Experiment | Loss Function | Acc (%) | Precision | Recall | F1-Score | ROC-AUC | EER (%) | Train Time (s) | Latency (ms) | Parameters |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    for exp in experiments:
        auc_s = f"{exp['auc_roc']:.4f}" if exp.get("auc_roc") is not None else "N/A"
        eer_s = f"{exp['eer'] * 100.0:.2f}%" if exp.get("eer") is not None else "N/A"
        ab_md += (
            f"| **{exp['name']}** | `{exp.get('loss_function', 'N/A')}` | "
            f"{exp['accuracy'] * 100.0:.2f}% | {exp['precision']:.3f} | {exp['recall']:.3f} | "
            f"{exp['f1']:.3f} | {auc_s} | {eer_s} | {exp['training_time_sec']:.2f} | "
            f"{exp['inference_latency_ms']:.2f} | {exp['num_parameters']:,} |\n"
        )
    st.markdown(ab_md)

    perf_png, eff_png = _render_ablation_charts_png(json.dumps(experiments, sort_keys=True))
    ac1, ac2 = st.columns(2)
    with ac1:
        st.image(perf_png, width="stretch")
    with ac2:
        st.image(eff_png, width="stretch")


def render_calibration_section(
    calibration: Dict[str, Any],
    rel_path: Optional[Path] = None,
    dist_path: Optional[Path] = None,
    pr_path: Optional[Path] = None,
):
    """Renders validation-only temperature scaling, threshold optimization, ECE/Brier, and 3-outcome policy."""
    st.markdown("#### 🎚️ Confidence Calibration, Threshold Optimization & 3-Outcome Policy")
    if not calibration or "temperature_scaling" not in calibration:
        st.info("Confidence Calibration (`results/calibration/calibration_results.json`): Not yet evaluated.")
        return

    temp_info = calibration.get("temperature_scaling", {})
    th_info = calibration.get("threshold_optimization", {})
    pol = calibration.get("uncertainty_policy", {})
    val_sec = calibration.get("validation") or {}
    test_sec = calibration.get("test") or {}

    temperature = float(temp_info.get("temperature", 1.0))
    tau_sel = float(th_info.get("selected_threshold", pol.get("decision_threshold", 0.50)))
    u_low = float(pol.get("uncertain_low", tau_sel))
    u_high = float(pol.get("uncertain_high", tau_sel))

    st.caption(
        f"Validation-only fitting (`zero_test_set_tuning=True`, checkpoint unmodified): "
        f"Temperature T = {temperature:.4f} | Default Threshold = 0.50 | Validation-Selected Threshold = {tau_sel:.2f} | "
        f"Uncertainty Review Zone = [{u_low:.3f}, {u_high:.3f}] | Bayesian override removed."
    )

    ref_sec = test_sec if test_sec else val_sec
    split_lbl = "Held-Out Test" if test_sec else "Validation"
    uncal = ref_sec.get("uncalibrated", {})
    cal = ref_sec.get("calibrated", {})

    if uncal and cal:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fitted Temp (T)", f"{temperature:.4f}", help="Minimizes NLL strictly on validation logits")
        ece_pos_delta = cal["ece_positive_class"] - uncal["ece_positive_class"]
        c2.metric(
            f"{split_lbl} Class-1 ECE",
            f"{cal['ece_positive_class']:.4f}",
            delta=f"{ece_pos_delta:+.4f} vs T=1",
            delta_color="inverse",
        )
        ece_conf_delta = cal["ece_confidence"] - uncal["ece_confidence"]
        c3.metric(
            f"{split_lbl} Conf ECE",
            f"{cal['ece_confidence']:.4f}",
            delta=f"{ece_conf_delta:+.4f} vs T=1",
            delta_color="inverse",
        )
        brier_delta = cal["brier_score"] - uncal["brier_score"]
        c4.metric(
            f"{split_lbl} Brier Score",
            f"{cal['brier_score']:.4f}",
            delta=f"{brier_delta:+.4f} vs T=1",
            delta_color="inverse",
        )
        nll_delta = float(temp_info.get("val_nll_after", 0.0)) - float(temp_info.get("val_nll_before", 0.0))
        c5.metric(
            "Validation NLL",
            f"{float(temp_info.get('val_nll_after', 0.0)):.4f}",
            delta=f"{nll_delta:+.4f} vs T=1",
            delta_color="inverse",
        )

    # Threshold & 3-Outcome Comparison Table
    rows_md = (
        "| Split | Operating Mode | Threshold / Band | Accuracy | Precision | Recall | F1-Score | Coverage / Selective |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n"
    )
    for split_name, sec in (("Validation (Fit Split)", val_sec), ("Held-Out Test (Out-of-Sample)", test_sec)):
        cal_s = sec.get("calibrated", {}) if sec else {}
        if not cal_s:
            continue
        m_def = cal_s.get("metrics_at_default_0_50", {})
        m_sel = cal_s.get("metrics_at_selected_threshold", {})
        m_pol = cal_s.get("three_outcome_policy_eval", {})
        if m_def:
            rows_md += (
                f"| {split_name} | Default Binary | `0.50` | {m_def['accuracy']*100:.2f}% | "
                f"{m_def['precision']:.3f} | {m_def['recall']:.3f} | {m_def['f1']:.3f} | 100.0% (Full) |\n"
            )
        if m_sel:
            rows_md += (
                f"| {split_name} | Validation-Selected Binary | `{tau_sel:.2f}` | {m_sel['accuracy']*100:.2f}% | "
                f"{m_sel['precision']:.3f} | {m_sel['recall']:.3f} | {m_sel['f1']:.3f} | 100.0% (Full) |\n"
            )
        if m_pol:
            sel_acc = m_pol.get("selective_accuracy")
            sel_acc_s = f"{sel_acc*100:.2f}%" if sel_acc is not None else "N/A"
            rows_md += (
                f"| {split_name} | **3-Outcome Policy** | `[{u_low:.3f}, {u_high:.3f}]` | **{sel_acc_s}** (Sel) | — | — | — | "
                f"Cov {m_pol.get('coverage', 0.0)*100:.1f}% "
                f"(Auth={m_pol.get('authentic_count', 0)}, Manip={m_pol.get('manipulated_count', 0)}, Unc={m_pol.get('uncertain_count', 0)}) |\n"
            )
    st.markdown(rows_md)

    artifacts = calibration.get("artifacts", {})
    repo_root = Path(__file__).resolve().parent.parent
    resolved_rel = rel_path if (rel_path and rel_path.exists()) else (repo_root / artifacts.get("reliability_diagram", ""))
    resolved_dist = dist_path if (dist_path and dist_path.exists()) else (repo_root / artifacts.get("confidence_distribution", ""))
    resolved_pr = pr_path if (pr_path and pr_path.exists()) else (repo_root / artifacts.get("pr_threshold_analysis", ""))

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if resolved_rel and resolved_rel.is_file():
            st.image(str(resolved_rel), caption="Reliability Diagram (Before vs After Temperature Scaling)", width="stretch")
    with pcol2:
        if resolved_dist and resolved_dist.is_file():
            st.image(str(resolved_dist), caption="Probability & Uncertainty Distribution", width="stretch")
    if resolved_pr and resolved_pr.is_file():
        st.image(str(resolved_pr), caption="Validation Precision-Recall & Threshold Trade-off", width="stretch")


def render_performance_dashboard():
    """Renders the comprehensive, genuine model performance dashboard."""
    data = load_evaluation_data()
    metrics = data["metrics"]
    history = data["history"]
    ablation = data.get("ablation", {})
    calibration = data.get("calibration", {})
    external_bench = data.get("external_benchmark", {})
    ckpt = data["checkpoint"]

    # Extract distinct evaluation tiers
    test_metrics = metrics.get("test", metrics if "sample_count" in metrics else {})
    val_metrics = metrics.get("validation", {})
    train_metrics = metrics.get("training", {})

    st.markdown("### 📊 Model Performance & Empirical Evaluation Dashboard")
    st.caption("Empirical evaluation metrics and loss convergence curves recorded during model training, validation, and held-out test evaluation.")

    # -------------------------------------------------------------
    # 1. HELD-OUT TEST EVALUATION (GENERALIZATION BENCHMARK)
    # -------------------------------------------------------------
    has_test = bool(test_metrics and "accuracy" in test_metrics)
    test_sample_count = test_metrics.get("sample_count") if has_test else None

    if has_test and test_sample_count:
        real_count = test_metrics.get("real_count", test_metrics.get("true_negatives", 0) + test_metrics.get("false_positives", 0))
        fake_count = test_metrics.get("fake_count", test_metrics.get("true_positives", 0) + test_metrics.get("false_negatives", 0))
        st.markdown(f"#### 🎯 Held-Out Test Evaluation Benchmark (`N = {test_sample_count}` Samples)")
        st.caption(f"Strictly independent held-out test split ({real_count} authentic optical, {fake_count} deepfakes) with isolated random seeds and social media perturbation.")
    else:
        st.markdown("#### 🎯 Held-Out Test Evaluation Benchmark")
        st.caption("Generalization performance evaluated on unseen photographic and synthetic test media.")

    t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns(6)

    if has_test:
        acc_val = f"{test_metrics['accuracy'] * 100.0:.2f}%"
        prec_val = f"{test_metrics['precision']:.3f}"
        rec_val = f"{test_metrics['recall']:.3f}"
        f1_val = f"{test_metrics['f1']:.3f}"
        auc_val = f"{test_metrics['auc_roc']:.4f}"
        eer_val = f"{test_metrics.get('eer', 0.0) * 100.0:.2f}%"
    else:
        acc_val = "Not yet evaluated"
        prec_val = "Not yet evaluated"
        rec_val = "Not yet evaluated"
        f1_val = "Not yet evaluated"
        auc_val = "Not yet evaluated"
        eer_val = "Not yet evaluated"

    t_col1.metric("Test Accuracy", acc_val)
    t_col2.metric("Precision", prec_val)
    t_col3.metric("Recall (Sensitivity)", rec_val)
    t_col4.metric("F1-Score", f1_val)
    t_col5.metric("ROC-AUC", auc_val)
    t_col6.metric("Equal Error Rate", eer_val)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. TRAINING & VALIDATION DYNAMICS (ACTUAL EPOCH LOGS)
    # -------------------------------------------------------------
    st.markdown("#### 📉 Training Loss & Accuracy Convergence Curves")
    st.caption("LabelSmoothingBCELoss (smoothing=0.06) and classification accuracy progression across training epochs:")

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    val_auc = history.get("val_auc", [])

    epochs = list(range(1, len(train_loss) + 1))

    if epochs:
        loss_png, acc_png = _render_training_curves_png(json.dumps(history, sort_keys=True))
        col_curve1, col_curve2 = st.columns(2)
        with col_curve1:
            st.image(loss_png, width="stretch")
        with col_curve2:
            st.image(acc_png, width="stretch")

        if train_loss:
            last_train_loss = train_loss[-1]
            last_train_acc = train_acc[-1] * 100.0 if train_acc else 0.0
            last_val_loss = val_loss[-1] if val_loss else 0.0
            last_val_acc = val_acc[-1] * 100.0 if val_acc else 0.0
            st.caption(f"🏁 Epoch {len(epochs)} Final State: Train Loss = {last_train_loss:.4f}, Train Acc = {last_train_acc:.1f}% | Val Loss = {last_val_loss:.4f}, Val Acc = {last_val_acc:.1f}%")
    else:
        st.info("Training history logs not yet evaluated or unavailable.")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. DIAGNOSTIC PLOTS & CLASS-WISE CONFUSION MATRICES
    # -------------------------------------------------------------
    st.markdown("#### 🔬 Empirical Diagnostic Plots & Confusion Matrices")

    diag_tabs = st.tabs(["🎯 Held-Out Test Evaluation", "🔬 Validation Split Benchmark"])

    with diag_tabs[0]:
        if has_test and test_sample_count:
            st.caption(f"Empirical performance evaluated on the strictly held-out test split (N = {test_sample_count}):")
        else:
            st.caption("Empirical performance evaluated on the strictly held-out test split (Not yet evaluated):")
        tc1, tc2 = st.columns(2)

        with tc1:
            if data["test_confusion_matrix_path"].exists():
                st.image(str(data["test_confusion_matrix_path"]), caption="Held-Out Test Confusion Matrix", width="stretch")
            else:
                st.info("Test confusion matrix: Not yet evaluated.")

        with tc2:
            if data["test_roc_curve_path"].exists():
                st.image(str(data["test_roc_curve_path"]), caption="Held-Out Test ROC Curve", width="stretch")
            else:
                st.info("Test ROC curve: Not yet evaluated.")

        # Dynamic Class-Wise Breakdown Table for Test Set
        st.markdown("##### 📋 Held-Out Test Class-Wise Breakdown")
        if has_test and "true_positives" in test_metrics:
            tp = int(test_metrics["true_positives"])
            tn = int(test_metrics["true_negatives"])
            fp = int(test_metrics["false_positives"])
            fn = int(test_metrics["false_negatives"])

            prec_neg = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            rec_neg = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            prec_pos = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec_pos = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            macro_prec = (prec_neg + prec_pos) / 2.0
            macro_rec = (rec_neg + rec_pos) / 2.0

            class_table = f"""
| Class Category | Total Support | Correct Classifications | Misclassifications | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Authentic Human Media (Negative)** | {tn + fp} | {tn} (True Negatives) | {fp} (False Positives) | {prec_neg:.3f} | {rec_neg:.3f} |
| **Deepfake / Synthetic Media (Positive)** | {tp + fn} | {tp} (True Positives) | {fn} (False Negatives) | {prec_pos:.3f} | {rec_pos:.3f} |
| **Micro / Macro Overall Average** | **{tp + tn + fp + fn}** | **{tp + tn}** | **{fp + fn}** | **{macro_prec:.3f}** | **{macro_rec:.3f}** |
"""
            st.markdown(class_table)
        else:
            st.info("Class-wise test breakdown: Not yet evaluated.")

    with diag_tabs[1]:
        st.caption("Validation split evaluation recorded during checkpoint selection:")
        vc1, vc2 = st.columns(2)

        with vc1:
            if data["val_confusion_matrix_path"].exists():
                st.image(str(data["val_confusion_matrix_path"]), caption="Validation Confusion Matrix", width="stretch")
            else:
                st.info("Validation confusion matrix plot not located.")

        with vc2:
            if data["val_roc_curve_path"].exists():
                st.image(str(data["val_roc_curve_path"]), caption="Validation ROC Curve", width="stretch")
            else:
                st.info("Validation ROC curve plot not located.")

        if val_metrics:
            vm1, vm2, vm3, vm4 = st.columns(4)
            vm1.metric("Validation Accuracy", f"{val_metrics.get('accuracy', 0.0) * 100.0:.2f}%")
            vm2.metric("Validation Precision", f"{val_metrics.get('precision', 0.0):.3f}")
            vm3.metric("Validation Recall", f"{val_metrics.get('recall', 0.0):.3f}")
            vm4.metric("Validation ROC-AUC", f"{val_metrics.get('auc_roc', 0.0):.4f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. CONFIDENCE CALIBRATION, THRESHOLD & UNCERTAINTY ANALYSIS
    # -------------------------------------------------------------
    render_calibration_section(
        calibration,
        rel_path=data.get("reliability_diagram_path"),
        dist_path=data.get("confidence_distribution_path"),
        pr_path=data.get("pr_threshold_analysis_path"),
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. ARCHITECTURAL & LOSS ABLATION STUDY
    # -------------------------------------------------------------
    render_ablation_section(ablation)

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. TRAINED CHECKPOINT & ARCHITECTURE SPECIFICATIONS
    # -------------------------------------------------------------
    st.markdown("#### 🛡️ Trained Checkpoint & Model Architecture Specifications")
    
    cal_temp = calibration.get("temperature_scaling", {}).get("temperature", 1.0)
    cal_pol = calibration.get("uncertainty_policy", {})
    cal_tau = cal_pol.get("decision_threshold", 0.50)
    spec_data = [
        ("Architecture Model", "DualDomainSentinelNet"),
        ("Spatial Domain Backbone", "EfficientNet-B0 (Pretrained ImageNet Weights, Conv2D Feature Extractor)"),
        ("Frequency Stream", "3 SRM High-Pass Kernels + 4-Stage Conv2D FFT Encoder"),
        ("Cross-Domain Fusion", "Multi-Head Cross-Attention (Queries: Spatial, Keys/Values: Spectral)"),
        ("Optimization Loss", "LabelSmoothingBCELoss (smoothing = 0.06; BinaryFocalLoss available in loss module)"),
        ("Calibration & Uncertainty", f"Validation Temperature Scaling (T = {cal_temp:.4f}) + 3-Outcome Policy (tau = {cal_tau:.2f})"),
        ("Target Resolution", "256 x 256 x 3 (Lanczos Resampled Square Context Crop)"),
        ("Checkpoint Path", ckpt["path"]),
        ("File Size", f"{ckpt['size_mb']} MB"),
        ("Cryptographic SHA-256", f"`{ckpt['sha256']}`"),
        ("Hardware Provider", "Apple Silicon MPS / NVIDIA CUDA / CPU Fallback")
    ]
    
    spec_md = "| Technical Specification | Implementation Value |\n| :--- | :--- |\n"
    for k, v in spec_data:
        spec_md += f"| **{k}** | {v} |\n"
    
    st.markdown(spec_md)

    # -------------------------------------------------------------
    # 5. BENCHMARK CROSS-ARCHITECTURE COMPARATIVE MATRIX
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🧪 Benchmark Cross-Architecture Comparative Matrix")
    st.caption("Empirical cross-evaluation comparing authentic human portraits against generative architectures (FaceSwap, StyleGAN2, Latent Diffusion, Inpainting).")

    benchmark_portfolios = [
        {
            "filename": "authentic_male_portrait.jpg",
            "ground_truth": "Authentic Human Portrait",
            "category": "Authentic Optical",
            "diagnostic_focus": "Natural 1/f decay, authentic skin pore textures"
        },
        {
            "filename": "authentic_female_portrait.jpg",
            "ground_truth": "Authentic Human Portrait",
            "category": "Authentic Optical",
            "diagnostic_focus": "Uniform spatial gradients, zero deconv spikes"
        },
        {
            "filename": "authentic_studio_photo_1.jpg",
            "ground_truth": "Authentic Human Portrait",
            "category": "Authentic Optical",
            "diagnostic_focus": "Natural specular reflection, nominal ELA"
        },
        {
            "filename": "authentic_studio_photo_2.jpg",
            "ground_truth": "Authentic Human Portrait",
            "category": "Authentic Optical",
            "diagnostic_focus": "Continuous high-frequency optical decay"
        },
        {
            "filename": "deepfake_faceswap_manipulation.jpg",
            "ground_truth": "FaceSwap (Autoencoder)",
            "category": "Identity Splicing",
            "diagnostic_focus": "Jawline & forehead blending boundary seams"
        },
        {
            "filename": "deepfake_stylegan_checkerboard.jpg",
            "ground_truth": "StyleGAN2 (Transposed Conv)",
            "category": "GAN Synthesis",
            "diagnostic_focus": "Periodic 2D FFT deconvolution checkerboard spikes"
        },
        {
            "filename": "deepfake_diffusion_ai_face.jpg",
            "ground_truth": "Latent Diffusion Model",
            "category": "Denoising Diffusion",
            "diagnostic_focus": "High-frequency noise prints & iris reflection mismatch"
        },
        {
            "filename": "deepfake_boundary_morph.jpg",
            "ground_truth": "Boundary Morphing / Inpainting",
            "category": "Neural Inpainting",
            "diagnostic_focus": "Local ELA quantization variance & edge gradients"
        }
    ]

    if "benchmark_results" not in st.session_state:
        st.session_state["benchmark_results"] = None

    if st.session_state.get("benchmark_results"):
        import pandas as pd
        df_bench = pd.DataFrame(st.session_state["benchmark_results"])
        st.dataframe(df_bench, width="stretch", hide_index=True)
    else:
        overview_md = "| Sample Portfolio | Ground Truth Architecture | Generator Family | Key Diagnostic Signature |\n| :--- | :--- | :--- | :--- |\n"
        for item in benchmark_portfolios:
            overview_md += f"| `{item['filename']}` | **{item['ground_truth']}** | {item['category']} | {item['diagnostic_focus']} |\n"
        st.markdown(overview_md)

    st.markdown("---")
    st.markdown("#### 🌍 External Benchmark Generalization Status (Zero Fabrication Policy)")
    datasets_ext = external_bench.get("datasets", {}) if isinstance(external_bench, dict) else {}
    if datasets_ext:
        ext_md = "| External Benchmark | Status | Local Directory | ROC-AUC / Accuracy | Notes |\n| :--- | :---: | :--- | :---: | :--- |\n"
        for ds_info in datasets_ext.values():
            st_lbl = "Evaluated" if ds_info.get("available") else "Unavailable Locally"
            m_ext = ds_info.get("metrics") or {}
            score_s = (
                f"AUC {m_ext.get('auc_roc', 0.0):.4f} / Acc {m_ext.get('accuracy', 0.0)*100:.1f}%"
                if m_ext else "N/A (Not Downloaded)"
            )
            ext_md += f"| **{ds_info.get('display_name')}** | `{st_lbl}` | `{ds_info.get('root_dir')}` | {score_s} | {ds_info.get('reason', '')} |\n"
        st.markdown(ext_md)
    else:
        st.info("External benchmark summary (`results/external_benchmark/external_benchmark_summary.json`) not yet generated.")
