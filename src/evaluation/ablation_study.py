"""
Controlled Architectural & Loss Ablation Study Runner.

Compares 5 configurations under equivalent StratifiedGroupKFold-disjoint data splits
and identical optimizer/seed/evaluation conditions:
1. Experiment 1: Spatial-only EfficientNet-B0 (BCEWithLogitsLoss)
2. Experiment 2: Frequency-only SRM + 2D FFT (BCEWithLogitsLoss)
3. Experiment 3: Spatial + Frequency with simple concatenation (BCEWithLogitsLoss)
4. Experiment 4: Spatial + Frequency with cross-attention fusion (BCEWithLogitsLoss)
5. Experiment 5: Full cross-attention model + existing training loss (LabelSmoothingBCELoss)

Measures Accuracy, Precision, Recall, F1-score, ROC-AUC, EER, training time,
end-to-end single-image inference latency, and exact parameter count.
Saves structured outputs to `results/ablation_results.json` without overwriting
the production checkpoint `checkpoints/best_model.pth`.
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.core.utils import get_device, load_config, set_seed
from src.data.dataset import DualDomainDataset, create_dataset_splits_from_dir
from src.evaluation.metrics import evaluate_predictions
from src.models.dual_domain_net import DualDomainSentinelNet
from src.models.loss import LabelSmoothingBCELoss


ABLATION_SPECS: List[Dict[str, Any]] = [
    {
        "experiment_id": "exp1_spatial_only",
        "experiment_number": 1,
        "name": "Experiment 1: Spatial-Only EfficientNet-B0",
        "short_label": "Exp 1\nSpatial Only",
        "fusion_type": "spatial_only",
        "loss_name": "BCEWithLogitsLoss",
        "use_existing_loss": False,
    },
    {
        "experiment_id": "exp2_frequency_only",
        "experiment_number": 2,
        "name": "Experiment 2: Frequency-Only SRM + FFT",
        "short_label": "Exp 2\nFreq Only",
        "fusion_type": "frequency_only",
        "loss_name": "BCEWithLogitsLoss",
        "use_existing_loss": False,
    },
    {
        "experiment_id": "exp3_concat_fusion",
        "experiment_number": 3,
        "name": "Experiment 3: Spatial + Frequency (Concatenation)",
        "short_label": "Exp 3\nConcat",
        "fusion_type": "concat",
        "loss_name": "BCEWithLogitsLoss",
        "use_existing_loss": False,
    },
    {
        "experiment_id": "exp4_cross_attention",
        "experiment_number": 4,
        "name": "Experiment 4: Spatial + Frequency (Cross-Attention)",
        "short_label": "Exp 4\nCross-Attn",
        "fusion_type": "cross_attention",
        "loss_name": "BCEWithLogitsLoss",
        "use_existing_loss": False,
    },
    {
        "experiment_id": "exp5_full_model_existing_loss",
        "experiment_number": 5,
        "name": "Experiment 5: Full Model + Existing Training Loss",
        "short_label": "Exp 5\nFull + RegLoss",
        "fusion_type": "cross_attention",
        "loss_name": "LabelSmoothingBCELoss(smoothing=0.06)",
        "use_existing_loss": True,
    },
]


def _sync_device(device: torch.device):
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _extract_split_tensors(
    dataset: DualDomainDataset,
    shared_backbone_features: Optional[nn.Module],
    shared_backbone_pool: Optional[nn.Module],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Loads spatial images, spectral FFT maps, labels, and (when spatial backbone exists)
    frozen EfficientNet-B0 pooled features once so every ablation experiment sees
    identical deterministic tensors across train, val, and test splits.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_spatial, all_spectral, all_labels, all_flat = [], [], [], []

    for batch in loader:
        sp = batch["spatial"]
        fr = batch["spectral"]
        lb = batch["label"]
        all_spatial.append(sp)
        all_spectral.append(fr)
        all_labels.append(lb)

        if shared_backbone_features is not None and shared_backbone_pool is not None:
            feat_map = shared_backbone_features(sp.to(device))
            pooled = shared_backbone_pool(feat_map)
            flat = torch.flatten(pooled, 1).cpu()
            all_flat.append(flat)

    return {
        "spatial": torch.cat(all_spatial, dim=0),
        "spectral": torch.cat(all_spectral, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "spatial_flat": torch.cat(all_flat, dim=0) if all_flat else torch.empty(0),
    }


def _train_single_ablation(
    spec: Dict[str, Any],
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    shared_spatial_state: Optional[Dict[str, torch.Tensor]],
    backbone_name: str,
    pretrained: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> Tuple[DualDomainSentinelNet, Dict[str, Any]]:
    """Trains one ablation variant fairly and returns best-val checkpoint + train stats."""
    set_seed(seed)

    model = DualDomainSentinelNet(
        backbone_name=backbone_name,
        pretrained=False if shared_spatial_state is not None else pretrained,
        spatial_dim=512,
        spectral_dim=256,
        fusion_type=spec["fusion_type"],
        dropout=0.3,
    ).to(device)

    # Share exact frozen backbone weights when spatial stream is enabled
    if model.spatial_stream is not None and shared_spatial_state is not None:
        model.spatial_stream.features.load_state_dict(shared_spatial_state)
        for param in model.spatial_stream.features.parameters():
            param.requires_grad = False
        model.spatial_stream.features.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)

    criterion: nn.Module = (
        LabelSmoothingBCELoss(smoothing=0.06)
        if spec["use_existing_loss"]
        else nn.BCEWithLogitsLoss()
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    is_freq_only = spec["fusion_type"] == "frequency_only"
    train_ds_t = TensorDataset(
        train_data["spectral"] if is_freq_only else train_data["spatial_flat"],
        train_data["spectral"],
        train_data["labels"],
    )
    val_ds_t = TensorDataset(
        val_data["spectral"] if is_freq_only else val_data["spatial_flat"],
        val_data["spectral"],
        val_data["labels"],
    )

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds_t, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds_t, batch_size=batch_size, shuffle=False)

    best_val_score = (-1.0, float("-inf"))  # (val_auc, -val_loss)
    best_state = copy.deepcopy(model.state_dict())
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    _sync_device(device)
    t_train_start = time.perf_counter()

    for _epoch in range(epochs):
        model.train()
        if model.spatial_stream is not None:
            model.spatial_stream.features.eval()

        running_loss = 0.0
        for first_b, spec_b, lbl_b in train_loader:
            spec_b = spec_b.to(device)
            lbl_b = lbl_b.to(device)

            optimizer.zero_grad()
            if is_freq_only:
                logits = model(spectral_tensor=spec_b)
            else:
                flat_b = first_b.to(device)
                logits = model(spatial_flat=flat_b, spectral_tensor=spec_b)

            loss = criterion(logits, lbl_b.view_as(logits))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * lbl_b.size(0)

        train_loss = running_loss / max(len(train_ds_t), 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for first_b, spec_b, lbl_b in val_loader:
                spec_b = spec_b.to(device)
                lbl_b = lbl_b.to(device)
                if is_freq_only:
                    logits = model(spectral_tensor=spec_b)
                else:
                    flat_b = first_b.to(device)
                    logits = model(spatial_flat=flat_b, spectral_tensor=spec_b)
                v_loss = criterion(logits, lbl_b.view_as(logits))
                val_loss_sum += float(v_loss.item()) * lbl_b.size(0)
                probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                val_probs.extend(probs.tolist())
                val_labels.extend(lbl_b.detach().cpu().numpy().reshape(-1).tolist())

        val_loss = val_loss_sum / max(len(val_ds_t), 1)
        val_y = np.asarray(val_labels, dtype=int)
        val_p = np.asarray(val_probs, dtype=float)
        val_acc = float(accuracy_score(val_y, (val_p >= 0.5).astype(int)))
        val_auc = float(roc_auc_score(val_y, val_p)) if len(np.unique(val_y)) >= 2 else 0.5

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))
        history["val_auc"].append(round(val_auc, 5))

        score_tuple = (val_auc, -val_loss)
        if score_tuple >= best_val_score:
            best_val_score = score_tuple
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    _sync_device(device)
    head_train_seconds = time.perf_counter() - t_train_start

    model.load_state_dict(best_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_dir / f"{spec['experiment_id']}.pth"
    torch.save(model.state_dict(), ckpt_file)

    return model, {
        "total_parameters": int(total_param_count),
        "trainable_parameters": int(trainable_param_count),
        "head_training_time_sec": round(head_train_seconds, 3),
        "best_val_auc": round(best_val_score[0], 5),
        "checkpoint_path": str(ckpt_file),
        "history": history,
    }


@torch.no_grad()
def _evaluate_end_to_end_test(
    model: DualDomainSentinelNet,
    spec: Dict[str, Any],
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
    latency_samples: int = 100,
    warmup: int = 8,
) -> Tuple[Dict[str, Any], float]:
    """
    Runs full end-to-end forward pass over held-out test images (un-cached tensors)
    to measure genuine accuracy/ROC-AUC/EER and true single-image inference latency.
    """
    model.eval()
    sp_all = test_data["spatial"]
    fr_all = test_data["spectral"]
    y_all = test_data["labels"].numpy().astype(int)

    n_test = len(y_all)
    y_scores: List[float] = []

    for start in range(0, n_test, batch_size):
        end = min(n_test, start + batch_size)
        sp_b = sp_all[start:end].to(device)
        fr_b = fr_all[start:end].to(device)
        logits = model(spatial_tensor=sp_b, spectral_tensor=fr_b)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        y_scores.extend(probs.tolist())

    metrics = evaluate_predictions(y_all, np.asarray(y_scores, dtype=float), threshold=0.50)
    metrics["sample_count"] = int(n_test)

    # Measure true single-image end-to-end latency on device
    n_bench = min(n_test, latency_samples)
    for w in range(min(warmup, n_bench)):
        sp_1 = sp_all[w : w + 1].to(device)
        fr_1 = fr_all[w : w + 1].to(device)
        _ = model(spatial_tensor=sp_1, spectral_tensor=fr_1)
    _sync_device(device)

    t0 = time.perf_counter()
    for i in range(n_bench):
        sp_1 = sp_all[i : i + 1].to(device)
        fr_1 = fr_all[i : i + 1].to(device)
        _ = model(spatial_tensor=sp_1, spectral_tensor=fr_1)
    _sync_device(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    mean_latency_ms = total_ms / max(n_bench, 1)

    return metrics, round(float(mean_latency_ms), 3)


def run_ablation_study(
    data_dir: str = "data",
    manifest_path: Optional[str] = "results/group_split_manifest.json",
    output_path: str = "results/ablation_results.json",
    ablation_ckpt_dir: str = "checkpoints/ablation",
    config_path: str = "config/default_config.yaml",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device_str: str = "auto",
    pretrained: Optional[bool] = None,
    max_samples_per_split: Optional[int] = None,
    latency_samples: int = 100,
) -> Dict[str, Any]:
    """
    Runs all 5 ablation experiments fairly on identical splits and saves results JSON.
    Never modifies `checkpoints/best_model.pth`.
    """
    cfg = load_config(config_path) if Path(config_path).exists() else {}
    backbone_name = cfg.get("model", {}).get("backbone", "efficientnet_b0")
    use_pretrained = cfg.get("model", {}).get("pretrained", True) if pretrained is None else pretrained

    device = get_device(device_str)
    set_seed(seed)

    # 1. Load group-disjoint splits cleanly
    use_manifest = manifest_path if (manifest_path and Path(manifest_path).exists()) else None
    train_ds, val_ds, test_ds = create_dataset_splits_from_dir(
        data_dir=data_dir,
        seed=seed,
        use_group_split=True,
        manifest_path=use_manifest,
    )

    if max_samples_per_split is not None:
        def _slice_ds(ds: DualDomainDataset, n_max: int) -> DualDomainDataset:
            return DualDomainDataset(
                ds.image_paths[:n_max],
                ds.labels[:n_max],
                is_train=ds.is_train,
                img_size=ds.img_size,
                source_groups=ds.source_groups[:n_max] if ds.source_groups else None,
            )
        train_ds = _slice_ds(train_ds, max_samples_per_split)
        val_ds = _slice_ds(val_ds, max_samples_per_split)
        test_ds = _slice_ds(test_ds, max_samples_per_split)

    # Disable random spatial jitter during deterministic shared tensor cache extraction
    # so Exp 1..5 receive 100% identical train/val/test tensor sets
    deterministic_train_ds = DualDomainDataset(
        train_ds.image_paths,
        train_ds.labels,
        is_train=False,
        img_size=train_ds.img_size,
        source_groups=train_ds.source_groups,
    )

    # 2. Extract shared pretrained SpatialBackbone once cleanly
    ref_model = DualDomainSentinelNet(
        backbone_name=backbone_name,
        pretrained=use_pretrained,
        fusion_type="spatial_only",
    ).to(device)
    assert ref_model.spatial_stream is not None
    ref_model.eval()
    shared_spatial_state = copy.deepcopy(ref_model.spatial_stream.features.state_dict())

    _sync_device(device)
    t_pre_start = time.perf_counter()
    train_data = _extract_split_tensors(
        deterministic_train_ds, ref_model.spatial_stream.features, ref_model.spatial_stream.pool, device, batch_size
    )
    val_data = _extract_split_tensors(
        val_ds, ref_model.spatial_stream.features, ref_model.spatial_stream.pool, device, batch_size
    )
    test_data = _extract_split_tensors(
        test_ds, None, None, device, batch_size
    )
    _sync_device(device)
    precompute_sec = time.perf_counter() - t_pre_start
    del ref_model

    # 3. Execute all 5 experiments under identical hyperparameters & data splits
    ckpt_dir = Path(ablation_ckpt_dir)
    experiments_out: List[Dict[str, Any]] = []

    for spec in ABLATION_SPECS:
        model, train_info = _train_single_ablation(
            spec=spec,
            train_data=train_data,
            val_data=val_data,
            shared_spatial_state=shared_spatial_state,
            backbone_name=backbone_name,
            pretrained=use_pretrained,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            checkpoint_dir=ckpt_dir,
        )
        test_metrics, latency_ms = _evaluate_end_to_end_test(
            model=model,
            spec=spec,
            test_data=test_data,
            device=device,
            batch_size=batch_size,
            latency_samples=latency_samples,
        )

        # Include feature extraction overhead fairly on spatial experiments
        spatial_overhead = precompute_sec if spec["fusion_type"] != "frequency_only" else 0.0
        total_train_time = train_info["head_training_time_sec"] + (spatial_overhead * 0.5)

        exp_record = {
            "experiment_id": spec["experiment_id"],
            "experiment_number": spec["experiment_number"],
            "name": spec["name"],
            "short_label": spec["short_label"],
            "fusion_type": spec["fusion_type"],
            "loss_function": spec["loss_name"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "auc_roc": test_metrics["auc_roc"],
            "pr_auc": test_metrics["pr_auc"],
            "eer": test_metrics["eer"],
            "fpr": test_metrics["fpr"],
            "fnr": test_metrics["fnr"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "training_time_sec": round(total_train_time, 3),
            "head_training_time_sec": train_info["head_training_time_sec"],
            "inference_latency_ms": latency_ms,
            "num_parameters": train_info["total_parameters"],
            "trainable_parameters": train_info["trainable_parameters"],
            "best_val_auc": train_info["best_val_auc"],
            "checkpoint_path": train_info["checkpoint_path"],
            "history": train_info["history"],
        }
        experiments_out.append(exp_record)

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "study_protocol": {
            "split_source": use_manifest or data_dir,
            "group_disjoint": bool(use_manifest is not None),
            "train_samples": int(len(train_ds)),
            "val_samples": int(len(val_ds)),
            "test_samples": int(len(test_ds)),
            "backbone": backbone_name,
            "pretrained_backbone": bool(use_pretrained),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "device": device.type.upper(),
            "shared_feature_extraction_sec": round(precompute_sec, 3),
            "note": (
                "All 5 experiments evaluate on identical group-disjoint splits using single-image "
                "end-to-end forward timing (batch=1) for inference latency."
            ),
        },
        "experiments": experiments_out,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Run 5-Experiment Architectural Ablation Study")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--manifest", type=str, default="results/group_split_manifest.json")
    parser.add_argument("--output", type=str, default="results/ablation_results.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    res = run_ablation_study(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_str=args.device,
    )
    print(f"[✓] Ablation Study Complete -> {args.output}")
    for exp in res["experiments"]:
        auc_str = f"{exp['auc_roc']:.4f}" if exp["auc_roc"] is not None else "N/A"
        eer_str = f"{exp['eer']*100:.2f}%" if exp["eer"] is not None else "N/A"
        print(
            f"    {exp['name']:48s} | Acc={exp['accuracy']*100:6.2f}% | "
            f"F1={exp['f1']:.3f} | AUC={auc_str} | EER={eer_str} | "
            f"Latency={exp['inference_latency_ms']:.2f}ms | Params={exp['num_parameters']:,}"
        )


if __name__ == "__main__":
    main()
