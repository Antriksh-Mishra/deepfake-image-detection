import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
except ImportError:
    accuracy_score = precision_score = recall_score = f1_score = roc_auc_score = confusion_matrix = None

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluation & Benchmark Engine for Deepfake Detection Models."""

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Runs evaluation over test DataLoader and computes metric suite."""
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 (Deepfake)
                _, preds = outputs.max(1)

                all_targets.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        if accuracy_score is not None and len(y_true) > 0:
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                roc_auc = 0.5
            cm = confusion_matrix(y_true, y_pred).tolist()
        else:
            # Fallback numpy calculation if scikit-learn not imported
            correct = (y_true == y_pred).sum()
            acc = float(correct / max(len(y_true), 1))
            prec = rec = f1 = roc_auc = acc
            cm = [[0, 0], [0, 0]]

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "total_samples": len(y_true)
        }

        return metrics

    @staticmethod
    def generate_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
        """Generates formatted Markdown metric comparison table."""
        table = "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
        table += "|---|---|---|---|---|---|\n"
        for name, m in results.items():
            table += f"| {name} | {m.get('accuracy', 0):.4f} | {m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | {m.get('f1_score', 0):.4f} | {m.get('roc_auc', 0):.4f} |\n"
        return table
