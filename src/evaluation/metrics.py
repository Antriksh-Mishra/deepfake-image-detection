"""
Comprehensive Academic Evaluation Metrics.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC,
False Positive Rate (FPR), False Negative Rate (FNR), Equal Error Rate (EER),
and produces publication-quality ROC, Precision-Recall, and Confusion Matrix plots.
"""

from typing import Any, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes Equal Error Rate (EER) and the corresponding operating threshold.
    EER is the point where False Acceptance Rate (FPR) equals False Rejection Rate (FNR).
    Returns (None, None) when only a single class is present.
    """
    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        return None, None

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_scores)
    fnr = 1.0 - tpr

    eer_idx = int(np.nanargmin(np.absolute(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes comprehensive binary forensic evaluation metrics at a fixed threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores_arr >= threshold).astype(int)

    acc = accuracy_score(y_true_arr, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred, average="binary", zero_division=0
    )

    has_both_classes = len(np.unique(y_true_arr)) >= 2
    if has_both_classes:
        auc: Optional[float] = float(roc_auc_score(y_true_arr, y_scores_arr))
        pr_auc: Optional[float] = float(average_precision_score(y_true_arr, y_scores_arr))
        eer, eer_thresh = compute_eer(y_true_arr, y_scores_arr)
    else:
        auc = None
        pr_auc = None
        eer, eer_thresh = None, None

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "fpr": fpr_val,
        "fnr": fnr_val,
        "specificity": specificity,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "threshold_used": float(threshold),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = "roc_curve.png",
    title: str = "Receiver Operating Characteristic (ROC) Curve",
):
    """Generates and saves publication-quality ROC curve with EER operating point."""
    y_true_arr = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true_arr)) < 2:
        plt.figure(figsize=(7, 6))
        plt.text(0.5, 0.5, "ROC Curve Unavailable\n(Requires Both Positive & Negative Classes)", ha="center", va="center", fontsize=11)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    fpr, tpr, _ = roc_curve(y_true_arr, y_scores)
    auc = roc_auc_score(y_true_arr, y_scores)
    eer, _ = compute_eer(y_true_arr, y_scores)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Dual-Domain Sentinel (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    if eer is not None:
        plt.scatter([eer], [1 - eer], color="#dc2626", s=80, zorder=5, label=f"EER Operating Point ({eer * 100:.2f}%)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = "pr_curve.png",
    title: str = "Precision-Recall Curve",
):
    """Generates and saves publication-quality Precision-Recall curve."""
    y_true_arr = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true_arr)) < 2:
        plt.figure(figsize=(7, 6))
        plt.text(0.5, 0.5, "PR Curve Unavailable\n(Requires Both Positive & Negative Classes)", ha="center", va="center", fontsize=11)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    precision, recall, _ = precision_recall_curve(y_true_arr, y_scores)
    ap = average_precision_score(y_true_arr, y_scores)
    baseline = float(np.mean(y_true_arr))

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#059669", lw=2.5, label=f"Sentinel PR (AP = {ap:.4f})")
    plt.axhline(y=baseline, color="#9ca3af", lw=1.5, linestyle="--", label=f"Positive Prevalence ({baseline:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "confusion_matrix.png",
    title: str = "Deepfake Detection Confusion Matrix",
):
    """Generates and saves a clean confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Authentic Real", "Deepfake"],
        yticklabels=["Authentic Real", "Deepfake"],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
