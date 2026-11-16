"""
Independent External Dataset Generalization Benchmark Runner.

Evaluates a trained checkpoint across FaceForensics++, Celeb-DF, and DFDC:
- Zero training or fine-tuning on external samples (`model.eval()` + `torch.no_grad()`).
- Zero test-set threshold tuning (evaluates strictly at pre-fixed `threshold`, default 0.50).
- Consistent preprocessing: optional FaceExtractor alignment + SPATIAL_TRANSFORM + 2D FFT SPECTRAL_TRANSFORM.
- Generates dataset-specific confusion matrices, ROC curves, and Precision-Recall curves
  plus JSON reports under `results/external_benchmark/`.
- Never fabricates results if external datasets are unavailable on disk.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from src.core.face_extractor import FaceExtractor
from src.core.utils import (
    SPATIAL_TRANSFORM,
    SPECTRAL_TRANSFORM,
    get_device,
    load_config,
)
from src.data.external_datasets import (
    ExternalDatasetInfo,
    ExternalSample,
    discover_external_datasets,
    extract_uniform_video_frames,
)
from src.evaluation.metrics import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet


def _load_model_for_external_eval(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[DualDomainSentinelNet, str]:
    """Loads checkpoint strictly in eval mode without altering any weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cfg = config.get("model", {})
    model = DualDomainSentinelNet(
        backbone_name=model_cfg.get("backbone", "efficientnet_b0"),
        pretrained=False,
        spatial_dim=model_cfg.get("spatial_feat_dim", 512),
        spectral_dim=model_cfg.get("spectral_feat_dim", 256),
        fusion_type=model_cfg.get("fusion_type", "cross_attention"),
        num_heads=model_cfg.get("num_heads", 4),
        dropout=model_cfg.get("dropout", 0.3),
    )

    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return model, sha


def _prepare_sample_tensors(
    sample: ExternalSample,
    face_extractor: Optional[FaceExtractor],
    spectral_analyzer: SpectralAnalyzer,
    frames_per_video: int = 8,
    align_faces: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts an image or video ExternalSample into paired (spatial, spectral) tensors
    using consistent forensic preprocessing.
    """
    frames_pil: List[Image.Image] = []
    if sample.media_type == "video":
        for _, frame_im in extract_uniform_video_frames(Path(sample.path), max_frames=frames_per_video):
            frames_pil.append(frame_im)
    else:
        with Image.open(sample.path) as raw_im:
            frames_pil.append(ImageOps.exif_transpose(raw_im).convert("RGB"))

    tensor_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for frame in frames_pil:
        if align_faces and face_extractor is not None:
            aligned, _, _ = face_extractor.extract_face(frame)
        else:
            aligned = frame.resize((256, 256), Image.Resampling.LANCZOS)

        spatial_t = SPATIAL_TRANSFORM(aligned)
        mag_spec, _ = spectral_analyzer.compute_fft_magnitude(aligned)
        spectral_t = SPECTRAL_TRANSFORM(Image.fromarray(mag_spec))
        tensor_pairs.append((spatial_t, spectral_t))

    return tensor_pairs


def evaluate_external_dataset(
    dataset_info: ExternalDatasetInfo,
    model: DualDomainSentinelNet,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.50,
    temperature: float = 1.0,
    frames_per_video: int = 8,
    align_faces: bool = True,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Runs strictly independent zero-tuning evaluation on one external dataset.
    For video samples, averages calibrated frame probabilities per clip so each clip
    contributes once to sample-level metrics while recording total frames evaluated.
    """
    dataset_out = output_dir / dataset_info.key
    if not dataset_info.available or not dataset_info.samples:
        return {
            "dataset_key": dataset_info.key,
            "display_name": dataset_info.display_name,
            "status": dataset_info.status,
            "available": False,
            "root_dir": dataset_info.root_dir,
            "reason": dataset_info.reason,
            "metrics": None,
            "artifacts": {},
        }

    dataset_out.mkdir(parents=True, exist_ok=True)
    face_extractor = FaceExtractor(target_size=(256, 256)) if align_faces else None
    spectral_analyzer = SpectralAnalyzer(target_size=256)

    y_true: List[int] = []
    y_scores: List[float] = []
    sub_cats: List[str] = []
    failed_samples: List[Dict[str, str]] = []
    total_frames_evaluated = 0
    t_start = time.time()

    model.eval()
    temp_val = max(float(temperature), 1e-4)

    for sample in dataset_info.samples:
        try:
            pairs = _prepare_sample_tensors(
                sample=sample,
                face_extractor=face_extractor,
                spectral_analyzer=spectral_analyzer,
                frames_per_video=frames_per_video,
                align_faces=align_faces,
            )
            if not pairs:
                failed_samples.append({"path": sample.path, "error": "No readable frames extracted"})
                continue

            clip_probs: List[float] = []
            with torch.no_grad():
                for start in range(0, len(pairs), batch_size):
                    chunk = pairs[start : start + batch_size]
                    sp_batch = torch.stack([p[0] for p in chunk]).to(device)
                    fr_batch = torch.stack([p[1] for p in chunk]).to(device)
                    logits = model(sp_batch, fr_batch).squeeze(-1) / temp_val
                    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                    clip_probs.extend(probs.tolist())

            total_frames_evaluated += len(clip_probs)
            sample_prob = float(np.mean(clip_probs))
            y_true.append(int(sample.label))
            y_scores.append(sample_prob)
            sub_cats.append(sample.sub_category)
        except Exception as exc:
            failed_samples.append({"path": sample.path, "error": f"{type(exc).__name__}: {exc}"})

    elapsed_s = time.time() - t_start
    if not y_true:
        return {
            "dataset_key": dataset_info.key,
            "display_name": dataset_info.display_name,
            "status": "failed",
            "available": True,
            "root_dir": dataset_info.root_dir,
            "reason": "Samples were discovered but all failed preprocessing/decoding.",
            "failed_samples": failed_samples,
            "metrics": None,
            "artifacts": {},
        }

    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    y_pred_arr = (y_scores_arr >= threshold).astype(int)

    metrics = evaluate_predictions(y_true_arr, y_scores_arr, threshold=threshold)
    metrics["sample_count"] = int(len(y_true_arr))
    metrics["real_count"] = int(np.sum(y_true_arr == 0))
    metrics["fake_count"] = int(np.sum(y_true_arr == 1))
    metrics["frames_evaluated"] = int(total_frames_evaluated)
    metrics["failed_sample_count"] = int(len(failed_samples))
    metrics["evaluation_seconds"] = round(elapsed_s, 3)

    # Sub-category accuracy breakdown (e.g., FF++ Deepfakes vs Face2Face vs NeuralTextures)
    sub_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(set(sub_cats)):
        mask = np.array([c == cat for c in sub_cats])
        cat_y = y_true_arr[mask]
        cat_pred = y_pred_arr[mask]
        sub_breakdown[cat] = {
            "count": int(np.sum(mask)),
            "accuracy": float(np.mean(cat_y == cat_pred)),
            "mean_fake_prob": float(np.mean(y_scores_arr[mask])),
        }

    cm_path = dataset_out / "confusion_matrix.png"
    roc_path = dataset_out / "roc_curve.png"
    pr_path = dataset_out / "pr_curve.png"

    plot_confusion_matrix(
        y_true_arr,
        y_pred_arr,
        save_path=str(cm_path),
        title=f"{dataset_info.display_name} Confusion Matrix (N={len(y_true_arr)})",
    )
    plot_roc_curve(
        y_true_arr,
        y_scores_arr,
        save_path=str(roc_path),
        title=f"{dataset_info.display_name} ROC Curve (Threshold-Free)",
    )
    plot_precision_recall_curve(
        y_true_arr,
        y_scores_arr,
        save_path=str(pr_path),
        title=f"{dataset_info.display_name} Precision-Recall Curve",
    )

    result_payload = {
        "dataset_key": dataset_info.key,
        "display_name": dataset_info.display_name,
        "status": "evaluated",
        "available": True,
        "root_dir": dataset_info.root_dir,
        "evaluation_policy": {
            "no_external_training": True,
            "no_test_threshold_tuning": True,
            "fixed_decision_threshold": float(threshold),
            "temperature": float(temperature),
            "align_faces": bool(align_faces),
            "frames_per_video": int(frames_per_video),
        },
        "metrics": metrics,
        "sub_category_breakdown": sub_breakdown,
        "failed_samples": failed_samples[:20],
        "artifacts": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
            "pr_curve": str(pr_path),
        },
    }

    metrics_file = output_dir / f"{dataset_info.key}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


def run_external_benchmarks(
    checkpoint_path: str = "checkpoints/best_model.pth",
    config_path: str = "config/default_config.yaml",
    output_dir: str = "results/external_benchmark",
    compression: str = "all",
    threshold: Optional[float] = None,
    temperature: float = 1.0,
    frames_per_video: Optional[int] = None,
    align_faces: Optional[bool] = None,
    max_per_class: Optional[int] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Discovers and independently evaluates all prioritized external datasets
    (FaceForensics++, Celeb-DF, DFDC), saving results cleanly in `results/external_benchmark/`.
    """
    cfg = load_config(config_path) if Path(config_path).exists() else {}
    ext_cfg = cfg.get("external_datasets", {})

    fixed_threshold = float(threshold if threshold is not None else ext_cfg.get("decision_threshold", 0.50))
    n_frames = int(frames_per_video if frames_per_video is not None else ext_cfg.get("frames_per_video", 8))
    use_align = bool(align_faces if align_faces is not None else ext_cfg.get("align_faces", True))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    discovered = discover_external_datasets(cfg, compression=compression, max_per_class=max_per_class)
    any_available = any(info.available for info in discovered.values())

    ckpt = Path(checkpoint_path)
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.exists() else "N/A"
    device = get_device(device_str)

    model = None
    if any_available:
        model, ckpt_sha = _load_model_for_external_eval(ckpt, cfg, device)

    results_by_dataset: Dict[str, Any] = {}
    for key, info in discovered.items():
        if info.available and model is not None:
            results_by_dataset[key] = evaluate_external_dataset(
                dataset_info=info,
                model=model,
                device=device,
                output_dir=out_path,
                threshold=fixed_threshold,
                temperature=temperature,
                frames_per_video=n_frames,
                align_faces=use_align,
            )
        else:
            results_by_dataset[key] = {
                "dataset_key": info.key,
                "display_name": info.display_name,
                "status": info.status,
                "available": False,
                "root_dir": info.root_dir,
                "reason": info.reason,
                "metrics": None,
                "artifacts": {},
            }

    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "device": device.type.upper(),
        "evaluation_guarantees": {
            "no_training_on_external_data": True,
            "no_test_set_threshold_tuning": True,
            "never_fabricates_unavailable_results": True,
            "fixed_threshold": fixed_threshold,
            "temperature": float(temperature),
        },
        "available_dataset_count": sum(1 for v in results_by_dataset.values() if v.get("available")),
        "datasets": results_by_dataset,
    }

    summary_file = out_path / "external_benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Independent External Dataset Benchmark Evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/external_benchmark")
    parser.add_argument("--compression", type=str, default="all", help="FF++ compression filter: raw, c23, c40, or all")
    parser.add_argument("--threshold", type=float, default=None, help="Pre-fixed threshold (not tuned on test set)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--frames_per_video", type=int, default=None)
    parser.add_argument("--no_align", action="store_true", help="Disable OpenCV face extraction if images are pre-cropped")
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    summary = run_external_benchmarks(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        compression=args.compression,
        threshold=args.threshold,
        temperature=args.temperature,
        frames_per_video=args.frames_per_video,
        align_faces=not args.no_align if args.no_align else None,
        max_per_class=args.max_per_class,
        device_str=args.device,
    )
    print(f"[✓] External Benchmark Summary Saved -> {Path(args.output_dir) / 'external_benchmark_summary.json'}")
    for key, ds in summary["datasets"].items():
        if ds["available"] and ds["metrics"]:
            m = ds["metrics"]
            print(
                f"    [{ds['display_name']}] N={m['sample_count']} | Acc={m['accuracy']*100:.2f}% | "
                f"F1={m['f1']:.3f} | AUC={m['auc_roc']:.4f} | EER={m['eer']*100:.2f}%"
            )
        else:
            print(f"    [{ds['display_name']}] status={ds['status']} — {ds['reason']}")


if __name__ == "__main__":
    main()
