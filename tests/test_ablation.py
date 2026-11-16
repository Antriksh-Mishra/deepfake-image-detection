"""
Unit & Integration Tests for Phase 4 Architectural & Loss Ablation Study.
Verifies all four DualDomainSentinelNet fusion modes, baseline checkpoint strict compatibility,
fair mini ablation execution on isolated temporary splits, `results/ablation_results.json` schema,
and dashboard ablation data loading.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.performance_dashboard import load_evaluation_data
from src.evaluation.ablation_study import ABLATION_SPECS, run_ablation_study
from src.models.dual_domain_net import DualDomainSentinelNet


@pytest.mark.parametrize(
    "fusion_type,expect_spatial,expect_freq",
    [
        ("spatial_only", True, False),
        ("frequency_only", False, True),
        ("concat", True, True),
        ("cross_attention", True, True),
    ],
)
def test_dual_domain_net_all_fusion_modes(fusion_type, expect_spatial, expect_freq):
    model = DualDomainSentinelNet(pretrained=False, fusion_type=fusion_type)
    assert (model.spatial_stream is not None) is expect_spatial
    assert (model.frequency_stream is not None) is expect_freq

    sp = torch.randn(2, 3, 256, 256)
    fr = torch.randn(2, 1, 256, 256)
    out = model(spatial_tensor=sp, spectral_tensor=fr)
    assert out.shape == (2, 1)

    if expect_spatial:
        flat = torch.randn(2, 1280)
        out_flat = model(spatial_flat=flat, spectral_tensor=fr)
        assert out_flat.shape == (2, 1)


def test_unsupported_fusion_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported fusion_type"):
        DualDomainSentinelNet(pretrained=False, fusion_type="invalid_fusion")


def test_production_checkpoint_strict_load_unaffected():
    ckpt_path = Path("checkpoints/best_model.pth")
    assert ckpt_path.exists()
    model = DualDomainSentinelNet(backbone_name="efficientnet_b0", pretrained=False, fusion_type="cross_attention")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert sum(p.numel() for p in model.parameters()) == 5_882_013


def _build_tiny_dataset(root: Path, n_per_class: int = 8):
    rng = np.random.RandomState(7)
    for split in ("train", "val", "test"):
        for cls_name, base_val in (("real", 190), ("fake", 75)):
            cls_dir = root / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(n_per_class):
                arr = np.clip(
                    base_val + rng.randint(-20, 20, size=(128, 128, 3)),
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(arr).save(cls_dir / f"{split}_{cls_name}_{idx}.jpg", quality=92)


def test_mini_ablation_study_end_to_end(tmp_path):
    data_dir = tmp_path / "mini_data"
    _build_tiny_dataset(data_dir, n_per_class=6)

    out_json = tmp_path / "ablation_results.json"
    ckpt_dir = tmp_path / "ablation_ckpts"

    res = run_ablation_study(
        data_dir=str(data_dir),
        manifest_path=None,
        output_path=str(out_json),
        ablation_ckpt_dir=str(ckpt_dir),
        epochs=2,
        batch_size=4,
        device_str="cpu",
        pretrained=False,
        max_samples_per_split=10,
        latency_samples=4,
    )

    assert out_json.exists()
    assert len(res["experiments"]) == len(ABLATION_SPECS) == 5

    required_keys = {
        "experiment_id",
        "experiment_number",
        "name",
        "fusion_type",
        "loss_function",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "eer",
        "training_time_sec",
        "inference_latency_ms",
        "num_parameters",
    }
    param_counts = {}
    for exp in res["experiments"]:
        assert required_keys.issubset(exp.keys())
        assert Path(exp["checkpoint_path"]).exists()
        assert 0.0 <= exp["accuracy"] <= 1.0
        assert exp["training_time_sec"] > 0.0
        assert exp["inference_latency_ms"] > 0.0
        assert exp["num_parameters"] > 0
        param_counts[exp["experiment_id"]] = exp["num_parameters"]

    # Parameter ordering sanity check
    assert param_counts["exp2_frequency_only"] < param_counts["exp1_spatial_only"]
    assert param_counts["exp1_spatial_only"] < param_counts["exp3_concat_fusion"]
    assert param_counts["exp3_concat_fusion"] < param_counts["exp4_cross_attention"]
    assert param_counts["exp4_cross_attention"] == param_counts["exp5_full_model_existing_loss"]


def test_real_ablation_results_and_dashboard_loader():
    results_file = Path("results/ablation_results.json")
    assert results_file.exists(), "Run `python -m src.evaluation.ablation_study` to generate ablation_results.json"
    payload = json.loads(results_file.read_text())

    assert payload["study_protocol"]["group_disjoint"] is True
    exps = payload["experiments"]
    assert len(exps) == 5
    for idx, exp in enumerate(exps, start=1):
        assert exp["experiment_number"] == idx
        for metric_key in ("accuracy", "precision", "recall", "f1", "auc_roc", "eer", "training_time_sec", "inference_latency_ms", "num_parameters"):
            assert exp[metric_key] is not None

    dash_data = load_evaluation_data()
    assert "ablation" in dash_data
    assert len(dash_data["ablation"].get("experiments", [])) == 5
