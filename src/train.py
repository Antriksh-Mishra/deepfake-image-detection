"""
Training Script for Dual-Domain Deepfake Sentinel Network.
Includes Apple Silicon MPS acceleration, ImageNet transfer learning with frozen
backbone features, trainable SRM frequency stream, trainable cross-attention fusion,
Focal Loss, Cosine Annealing LR schedule, validation checkpointing, and ROC-AUC tracking.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from src.core.utils import get_device, set_seed, load_config
from src.data.dataset import create_dataset_splits_from_dir
from src.models.dual_domain_net import DualDomainSentinelNet
from src.models.loss import BinaryFocalLoss, LabelSmoothingBCELoss


@torch.no_grad()
def precompute_tensors(
    model: DualDomainSentinelNet,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "train"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts spatial pooled features and bundles spectral tensors into memory.
    Avoids redundant forward passes through 81 frozen convolutional layers on each epoch.
    """
    model.eval()
    all_flat = []
    all_spectral = []
    all_labels = []
    
    print(f"[*] Pre-computing representations for {split_name} split ({len(dataloader.dataset)} samples)...", flush=True)
    for batch in dataloader:
        spatial = batch["spatial"].to(device)
        spectral = batch["spectral"]
        labels = batch["label"]

        feat_map = model.spatial_stream.features(spatial)
        pooled = model.spatial_stream.pool(feat_map)
        flat = torch.flatten(pooled, 1).cpu()

        all_flat.append(flat)
        all_spectral.append(spectral)
        all_labels.append(labels)

    flat_tensor = torch.cat(all_flat, dim=0)
    spectral_tensor = torch.cat(all_spectral, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return flat_tensor, spectral_tensor, labels_tensor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one training epoch and returns (mean_loss, accuracy)."""
    model.train()
    # Ensure backbone feature extractor stays in eval mode
    model.spatial_stream.features.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    for flat_b, spec_b, label_b in dataloader:
        flat_b = flat_b.to(device)
        spec_b = spec_b.to(device)
        label_b = label_b.to(device)

        optimizer.zero_grad()
        logits = model(spatial_flat=flat_b, spectral_tensor=spec_b)
        loss = criterion(logits, label_b)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * flat_b.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend((probs >= 0.5).astype(int).flatten())
        all_labels.extend(label_b.detach().cpu().numpy().astype(int).flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates model on validation set."""
    model.eval()
    running_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for flat_b, spec_b, label_b in dataloader:
            flat_b = flat_b.to(device)
            spec_b = spec_b.to(device)
            label_b = label_b.to(device)

            logits = model(spatial_flat=flat_b, spectral_tensor=spec_b)
            loss = criterion(logits, label_b)

            running_loss += loss.item() * flat_b.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_labels.extend(label_b.cpu().numpy().astype(int).flatten())

    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return {
        "val_loss": float(val_loss),
        "val_acc": float(acc),
        "val_precision": float(precision),
        "val_recall": float(recall),
        "val_f1": float(f1),
        "val_auc": float(auc)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Domain Deepfake Sentinel")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else {}
    set_seed(config.get("hardware", {}).get("seed", 42))
    device = get_device(config.get("hardware", {}).get("device", "auto"))
    print(f"[*] Training on device: {device} ({device.type.upper()})")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Raw Datasets
    print(f"[*] Loading data from: {args.data_dir}")
    raw_train_ds, raw_val_ds, raw_test_ds = create_dataset_splits_from_dir(args.data_dir)
    print(f"[✓] Train: {len(raw_train_ds)} | Val: {len(raw_val_ds)} | Test: {len(raw_test_ds)}")

    raw_train_loader = DataLoader(raw_train_ds, batch_size=32, shuffle=False)
    raw_val_loader = DataLoader(raw_val_ds, batch_size=32, shuffle=False)

    # 2. Instantiate Model
    model = DualDomainSentinelNet(
        backbone_name=config.get("model", {}).get("backbone", "efficientnet_b0"),
        pretrained=config.get("model", {}).get("pretrained", True),
        spatial_dim=config.get("model", {}).get("spatial_feat_dim", 512),
        spectral_dim=config.get("model", {}).get("spectral_feat_dim", 256),
        fusion_type=config.get("model", {}).get("fusion_type", "cross_attention")
    ).to(device)

    # Freeze ImageNet feature extraction layers for rapid & stable transfer learning
    for param in model.spatial_stream.features.parameters():
        param.requires_grad = False

    # 3. Pre-compute Cached Representations
    train_flat, train_spec, train_labels = precompute_tensors(model, raw_train_loader, device, "train")
    val_flat, val_spec, val_labels = precompute_tensors(model, raw_val_loader, device, "val")

    train_tensor_ds = TensorDataset(train_flat, train_spec, train_labels)
    val_tensor_ds = TensorDataset(val_flat, val_spec, val_labels)

    train_loader = DataLoader(train_tensor_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_ds, batch_size=args.batch_size, shuffle=False)

    # Trainable parameters: spatial projection, frequency stream, cross-attention fusion, classification head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[✓] Trainable parameters: {sum(p.numel() for p in trainable_params):,} weights")

    criterion = LabelSmoothingBCELoss(smoothing=0.06)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop
    best_val_auc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}

    print("\n" + "=" * 60)
    print(" STARTING DUAL-DOMAIN SENTINEL TRAINING SPRINT ")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["val_loss"]))
        history["val_acc"].append(float(val_metrics["val_acc"]))
        history["val_auc"].append(float(val_metrics["val_auc"]))
        history["val_f1"].append(float(val_metrics["val_f1"]))

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% || "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']*100:.1f}% | Val AUC: {val_metrics['val_auc']:.4f} | Val F1: {val_metrics['val_f1']:.4f}",
            flush=True
        )

        # Save best model checkpoint
        if val_metrics["val_auc"] >= best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  [★] Saved new best checkpoint (AUC: {best_val_auc:.4f}) -> {best_path}", flush=True)

        # Persist history after each epoch
        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[✓] Training complete! Best Validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
