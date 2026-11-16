"""
Checkpoint Evaluation Script for Dual-Domain Deepfake Sentinel Network.
Evaluates the trained checkpoint on:
1. Held-out test dataset (data/test)
2. Validation dataset (data/val)

Calculates genuine accuracy, precision, recall, F1-score, ROC-AUC, Equal Error Rate (EER),
and confusion matrices, generating publication-grade diagnostic plots and saving structured
metrics to checkpoints/evaluation_metrics.json.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.core.utils import (
    SPATIAL_TRANSFORM,
    SPECTRAL_TRANSFORM,
    compute_throughput_fps,
    get_device,
    safe_batched_forward,
)
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet
from src.evaluation.metrics import compute_eer, evaluate_predictions


def evaluate_split(
    model: torch.nn.Module,
    analyzer: SpectralAnalyzer,
    real_paths: List[Path],
    fake_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluates model on a specific split of real and fake images using batched forward inference."""
    all_paths = real_paths + fake_paths
    y_true = np.array([0] * len(real_paths) + [1] * len(fake_paths))
    y_scores = [0.5] * len(all_paths)

    if not all_paths:
        metrics = evaluate_predictions(y_true, np.array(y_scores))
        metrics['sample_count'] = 0
        metrics['real_count'] = 0
        metrics['fake_count'] = 0
        metrics['throughput_fps'] = 0.0
        return metrics, y_true, np.array(y_scores)

    t0 = time.time()
    model.eval()
    eff_batch = max(1, int(batch_size))
    active_dev = device

    for start in range(0, len(all_paths), eff_batch):
        chunk = all_paths[start:start + eff_batch]
        valid_indices: List[int] = []
        spatial_list: List[torch.Tensor] = []
        spectral_list: List[torch.Tensor] = []

        for offset, p in enumerate(chunk):
            global_idx = start + offset
            try:
                with Image.open(p) as raw_im:
                    img = raw_im.convert('RGB')
                sp_t = SPATIAL_TRANSFORM(img)
                mag_spec, _ = analyzer.compute_fft_magnitude(img)
                fr_t = SPECTRAL_TRANSFORM(Image.fromarray(mag_spec))
                spatial_list.append(sp_t)
                spectral_list.append(fr_t)
                valid_indices.append(global_idx)
            except Exception as e:
                print(f'[!] Error processing {p}: {e}')
                y_scores[global_idx] = 0.5

        if spatial_list:
            sp_batch = torch.stack(spatial_list)
            fr_batch = torch.stack(spectral_list)
            with torch.no_grad():
                logits, active_dev, _ = safe_batched_forward(
                    model,
                    sp_batch,
                    fr_batch,
                    active_dev,
                    enable_cpu_fallback=True,
                )
                probs = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy().tolist()
            for idx_pos, prob in zip(valid_indices, probs):
                y_scores[idx_pos] = float(prob)

    elapsed_ms = (time.time() - t0) * 1000.0
    y_scores_arr = np.array(y_scores)
    metrics = evaluate_predictions(y_true, y_scores_arr)
    metrics['sample_count'] = int(len(y_true))
    metrics['real_count'] = int(len(real_paths))
    metrics['fake_count'] = int(len(fake_paths))
    metrics['throughput_fps'] = compute_throughput_fps(len(y_true), elapsed_ms)

    return metrics, y_true, y_scores_arr


def plot_roc_with_eer(y_true: np.ndarray, y_scores: np.ndarray, save_path: str, title: str):
    """Generates high-resolution ROC curve with EER operating point."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    eer, _ = compute_eer(y_true, y_scores)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr, color='#2563eb', lw=2.4, label=f'Sentinel ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Chance (AUC = 0.5000)')
    plt.scatter([eer], [1.0 - eer], color='#dc2626', s=70, zorder=5, label=f'EER Operating Point ({eer*100.0:.2f}%)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_custom_confusion_matrix(cm: np.ndarray, save_path: str, title: str):
    """Generates clean confusion matrix heatmap with real counts."""
    plt.figure(figsize=(5.5, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Authentic Real', 'Deepfake'],
        yticklabels=['Authentic Real', 'Deepfake'],
        cbar=False,
        annot_kws={'size': 13, 'weight': 'bold'}
    )
    plt.xlabel('Predicted Class', fontsize=11)
    plt.ylabel('Ground Truth Class', fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_full_evaluation():
    repo_root = Path(__file__).resolve().parent.parent.parent
    checkpoints_dir = repo_root / 'checkpoints'
    data_dir = repo_root / 'data'
    ckpt_path = checkpoints_dir / 'best_model.pth'

    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found at {ckpt_path}')

    device = get_device('auto')
    print(f'[*] Evaluating checkpoint {ckpt_path.name} on device: {device}')

    model = DualDomainSentinelNet()
    loaded_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = loaded_ckpt["state_dict"] if isinstance(loaded_ckpt, dict) and "state_dict" in loaded_ckpt else loaded_ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    analyzer = SpectralAnalyzer(256)

    # 1. Evaluate on Held-Out Test Set
    test_real = sorted(list((data_dir / 'test' / 'real').glob('*.jpg')))
    test_fake = sorted(list((data_dir / 'test' / 'fake').glob('*.jpg')))
    print(f'[*] Evaluating Held-Out Test Split ({len(test_real)} Real, {len(test_fake)} Fake)...')
    test_metrics, test_y_true, test_y_scores = evaluate_split(model, analyzer, test_real, test_fake, device)

    # 2. Evaluate on Validation Set
    val_real = sorted(list((data_dir / 'val' / 'real').glob('*.jpg')))
    val_fake = sorted(list((data_dir / 'val' / 'fake').glob('*.jpg')))
    print(f'[*] Evaluating Validation Split ({len(val_real)} Real, {len(val_fake)} Fake)...')
    val_metrics, val_y_true, val_y_scores = evaluate_split(model, analyzer, val_real, val_fake, device)

    # 3. Read Training History if exists
    training_summary = {}
    history_path = checkpoints_dir / 'training_history.json'
    if history_path.exists():
        try:
            hist = json.loads(history_path.read_text())
            training_summary = {
                'epochs': len(hist.get('train_loss', [])),
                'final_train_loss': float(hist.get('train_loss', [-1])[-1]),
                'final_train_acc': float(hist.get('train_acc', [-1])[-1]),
                'final_val_loss': float(hist.get('val_loss', [-1])[-1]),
                'final_val_auc': float(hist.get('val_auc', [-1])[-1]),
            }
        except Exception as e:
            print(f'[!] Could not parse training history: {e}')

    # 4. Generate Publication-Quality Plots
    print('[*] Generating publication plots...')
    test_cm = np.array([
        [test_metrics['true_negatives'], test_metrics['false_positives']],
        [test_metrics['false_negatives'], test_metrics['true_positives']]
    ])
    val_cm = np.array([
        [val_metrics['true_negatives'], val_metrics['false_positives']],
        [val_metrics['false_negatives'], val_metrics['true_positives']]
    ])

    plot_roc_with_eer(test_y_true, test_y_scores, str(checkpoints_dir / 'test_roc_curve.png'), f'Held-Out Test ROC (AUC = {test_metrics["auc_roc"]:.4f})')
    plot_custom_confusion_matrix(test_cm, str(checkpoints_dir / 'test_confusion_matrix.png'), f'Held-Out Test Confusion Matrix (N = {test_metrics["sample_count"]})')

    plot_roc_with_eer(val_y_true, val_y_scores, str(checkpoints_dir / 'roc_curve.png'), f'Validation ROC (AUC = {val_metrics["auc_roc"]:.4f})')
    plot_custom_confusion_matrix(val_cm, str(checkpoints_dir / 'confusion_matrix.png'), f'Validation Confusion Matrix (N = {val_metrics["sample_count"]})')

    # 5. Save structured evaluation_metrics.json
    payload = {
        **test_metrics,  # Top-level backward compatible with test evaluation
        'test': test_metrics,
        'validation': val_metrics,
        'training': training_summary,
    }

    out_json = checkpoints_dir / 'evaluation_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'[✓] Successfully evaluated and saved to {out_json}!')
    print(f'    Test Set (N={test_metrics["sample_count"]}): Acc: {test_metrics["accuracy"]*100.0:.2f}%, AUC: {test_metrics["auc_roc"]:.4f}, EER: {test_metrics["eer"]*100.0:.2f}%')
    print(f'    Validation (N={val_metrics["sample_count"]}): Acc: {val_metrics["accuracy"]*100.0:.2f}%, AUC: {val_metrics["auc_roc"]:.4f}, EER: {val_metrics["eer"]*100.0:.2f}%')


if __name__ == '__main__':
    run_full_evaluation()