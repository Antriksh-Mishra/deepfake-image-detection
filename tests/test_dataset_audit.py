"""
Unit & Integration Tests for Phase 2 Dataset Integrity, Duplicate Detection,
Perceptual Hashing (pHash), Source-Group Clustering, and Leakage Prevention.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.data.dataset import DualDomainDataset, create_dataset_splits_from_dir
from src.data.dataset_audit import (
    compute_phash,
    phash_hamming_distance,
    run_dataset_audit,
    stratified_group_train_val_test_split,
)


def _make_pattern_image(seed: int, size: int = 128) -> Image.Image:
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (size, size), color=(int(rng.randint(20, 200)), int(rng.randint(20, 200)), int(rng.randint(20, 200))))
    draw = ImageDraw.Draw(img)
    for _ in range(5):
        x0 = int(rng.randint(0, size // 2))
        y0 = int(rng.randint(0, size // 2))
        x1 = int(rng.randint(size // 2, size))
        y1 = int(rng.randint(size // 2, size))
        col = (int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255)))
        draw.rectangle([x0, y0, x1, y1], fill=col)
    return img


def test_phash_stability_and_separation():
    img_a = _make_pattern_image(seed=10)
    img_b = _make_pattern_image(seed=99)

    hex_a, bits_a = compute_phash(img_a)
    _, bits_b = compute_phash(img_b)
    assert len(hex_a) == 16
    assert bits_a.shape == (64,)
    assert phash_hamming_distance(bits_a, bits_a) == 0
    assert phash_hamming_distance(bits_a, bits_b) >= 4

    # Slight brightness tweak stays very close in pHash space
    arr = np.clip(np.asarray(img_a, dtype=np.float32) * 1.03, 0, 255).astype(np.uint8)
    _, bits_a_near = compute_phash(Image.fromarray(arr))
    assert phash_hamming_distance(bits_a, bits_a_near) <= 4


def test_audit_detects_exact_near_corrupt_missing_non_destructively(tmp_path):
    data_dir = tmp_path / "ds"
    for split in ("train", "val", "test"):
        for cls in ("real", "fake"):
            (data_dir / split / cls).mkdir(parents=True)

    # Distinct synthetic source portraits per group
    img1 = _make_pattern_image(seed=1, size=128)
    img2 = _make_pattern_image(seed=2, size=128)
    img3 = _make_pattern_image(seed=3, size=128)

    p1 = data_dir / "train" / "real" / "000_001_f1.png"
    img1.save(p1)

    # Exact SHA-256 copy across split & inverted label
    p1_dup = data_dir / "test" / "fake" / "dup_copy.png"
    p1_dup.write_bytes(p1.read_bytes())

    # Near duplicate (slight pixel perturbation) in val
    arr_near = np.clip(np.asarray(img2, dtype=np.int16) + 2, 0, 255).astype(np.uint8)
    img2.save(data_dir / "train" / "real" / "id5_real.jpg", quality=95)
    Image.fromarray(arr_near).save(data_dir / "val" / "fake" / "id5_near.jpg", quality=95)

    # Disjoint test real sample
    img3.save(data_dir / "test" / "real" / "unique_test.jpg", quality=90)

    # Corrupted image file (must be counted, NOT deleted)
    corrupt_path = data_dir / "val" / "real" / "corrupt_broken.jpg"
    corrupt_path.write_bytes(b"NOT_A_VALID_JPEG_STREAM")

    audit_out = tmp_path / "results" / "dataset_audit.json"
    manifest_out = tmp_path / "results" / "manifest.json"
    report = run_dataset_audit(
        data_dir=str(data_dir),
        output_path=str(audit_out),
        manifest_output_path=str(manifest_out),
        expected_manifest_paths=["train/real/missing_image.jpg"],
    )

    # Ensure zero destructive modifications happened
    assert corrupt_path.exists()
    assert p1_dup.exists()

    assert audit_out.exists()
    assert report["total_image_count"] == 6
    assert report["valid_image_count"] == 5
    assert report["corrupted_image_count"] == 1
    assert report["missing_file_count"] == 1
    assert report["duplicate_image_count"] == 1
    assert report["exact_duplicate_analysis"]["cross_split_duplicate_group_count"] == 1
    assert report["exact_duplicate_analysis"]["cross_label_conflict_group_count"] == 1
    assert report["near_duplicate_image_count"] >= 4
    assert report["resolution_distribution"] == {"128x128": 5}
    assert report["leakage_detected"] is True
    assert report["requires_explicit_confirmation_for_cleanup"] is True


def test_stratified_group_kfold_no_leakage():
    # 6 distinct groups, 4 samples per group (both classes represented)
    indices = list(range(24))
    labels = [0, 1] * 12
    groups = [f"grp_{i // 4}" for i in range(24)]

    tr, va, te = stratified_group_train_val_test_split(indices, labels, groups, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert len(tr) + len(va) + len(te) == 24
    g_tr = {groups[i] for i in tr}
    g_va = {groups[i] for i in va}
    g_te = {groups[i] for i in te}
    assert not (g_tr & g_va)
    assert not (g_tr & g_te)
    assert not (g_va & g_te)

    with pytest.raises(ValueError):
        stratified_group_train_val_test_split([0, 1, 2, 3], [0, 1, 0, 1], ["g1", "g1", "g2", "g2"])


def test_dataset_corrupt_tracking_and_group_split(tmp_path):
    unified = tmp_path / "unified"
    (unified / "real").mkdir(parents=True)
    (unified / "fake").mkdir(parents=True)

    # Build 6 distinct source groups with real + fake variations sharing group manifest
    sg = {}
    for g in range(6):
        im_r = _make_pattern_image(seed=100 + g, size=64)
        im_f = _make_pattern_image(seed=200 + g, size=64)
        rp = unified / "real" / f"r_{g}.jpg"
        fp = unified / "fake" / f"f_{g}.jpg"
        im_r.save(rp)
        im_f.save(fp)
        sg[f"real/r_{g}.jpg"] = f"source_{g}"
        sg[f"fake/f_{g}.jpg"] = f"source_{g}"
    (unified / "source_groups.json").write_text(json.dumps(sg))

    tr_ds, va_ds, te_ds = create_dataset_splits_from_dir(str(unified), use_group_split=True, img_size=64)
    assert len(tr_ds) + len(va_ds) + len(te_ds) == 12
    assert not (set(tr_ds.source_groups) & set(va_ds.source_groups))
    assert not (set(tr_ds.source_groups) & set(te_ds.source_groups))
    assert not (set(va_ds.source_groups) & set(te_ds.source_groups))

    # Test corrupt tracking does not fail silently
    bad_img = tmp_path / "bad.jpg"
    bad_img.write_bytes(b"corrupt")
    ds = DualDomainDataset([str(bad_img)], [0], is_train=False, img_size=64)
    sample = ds[0]
    assert sample["is_corrupt"] is True
    assert str(bad_img) in ds.corrupted_paths

    # Empty test dir raises ValueError rather than aliasing validation
    presplit = tmp_path / "presplit"
    for d in ("train/real", "train/fake", "val/real", "val/fake", "test/real", "test/fake"):
        (presplit / d).mkdir(parents=True)
    _make_pattern_image(7, 64).save(presplit / "train/real/a.jpg")
    _make_pattern_image(8, 64).save(presplit / "train/fake/b.jpg")
    _make_pattern_image(9, 64).save(presplit / "val/real/c.jpg")
    _make_pattern_image(10, 64).save(presplit / "val/fake/d.jpg")
    with pytest.raises(ValueError):
        create_dataset_splits_from_dir(str(presplit), img_size=64)


def test_real_dataset_audit_artifact_exists():
    audit_path = Path("results/dataset_audit.json")
    assert audit_path.exists()
    data = json.loads(audit_path.read_text())
    for key in (
        "total_image_count",
        "real_fake_class_distribution",
        "train_val_test_distribution",
        "duplicate_image_count",
        "near_duplicate_image_count",
        "source_group_analysis",
        "resolution_distribution",
        "corrupted_image_count",
        "missing_file_count",
    ):
        assert key in data
    assert data["total_image_count"] == 2400
    assert data["corrupted_image_count"] == 0
    assert data["requires_explicit_confirmation_for_cleanup"] is True
