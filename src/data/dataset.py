"""
Dual-Domain PyTorch Dataset.
Loads face images and simultaneously prepares:
1. Spatial RGB tensor (normalized with ImageNet statistics).
2. Frequency 2D FFT magnitude spectrum tensor.
3. Binary ground truth label (0: Real, 1: Fake).

Supports group-aware StratifiedGroupKFold splitting via source-group provenance
and perceptual hashing, plus explicit corrupted/missing file tracking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms

from src.explainability.spectral_analysis import SpectralAnalyzer
from src.core.utils import SPATIAL_TRANSFORM, SPECTRAL_TRANSFORM
from src.data.dataset_audit import (
    IMAGE_EXTENSIONS,
    assign_source_groups,
    detect_near_duplicates,
    scan_dataset_images,
    stratified_group_train_val_test_split,
)

logger = logging.getLogger(__name__)


class DualDomainDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        labels: Sequence[int],
        is_train: bool = True,
        img_size: int = 256,
        source_groups: Optional[Sequence[str]] = None,
        raise_on_corrupt: bool = False,
    ):
        """
        Args:
            image_paths: Filepaths to face images.
            labels: Corresponding binary labels (0 for Real, 1 for Fake).
            is_train: Whether to apply training data augmentation.
            img_size: Target image dimension (default: 256x256).
            source_groups: Optional source-group identifiers aligned with image_paths.
            raise_on_corrupt: If True, raises RuntimeError on unreadable images rather
                than recording & returning a zero fallback image.
        """
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.source_groups = list(source_groups) if source_groups is not None else None
        self.is_train = is_train
        self.img_size = img_size
        self.raise_on_corrupt = raise_on_corrupt
        self.corrupted_paths: List[str] = []
        self.is_val_fallback: bool = False
        self.spectral_analyzer = SpectralAnalyzer(target_size=img_size)

        if is_train:
            self.spatial_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.spatial_transform = SPATIAL_TRANSFORM

        self.spectral_transform = SPECTRAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        is_corrupt = False

        try:
            with Image.open(img_path) as raw_im:
                image = ImageOps.exif_transpose(raw_im).convert("RGB")
        except Exception as exc:
            is_corrupt = True
            if img_path not in self.corrupted_paths:
                self.corrupted_paths.append(img_path)
            logger.warning("Corrupted or unreadable dataset image %s: %s", img_path, exc)
            if self.raise_on_corrupt:
                raise RuntimeError(f"Corrupted dataset image: {img_path}") from exc
            image = Image.new("RGB", (self.img_size, self.img_size), color=(0, 0, 0))

        spatial_tensor = self.spatial_transform(image)
        mag_spec, _ = self.spectral_analyzer.compute_fft_magnitude(image)
        spec_img = Image.fromarray(mag_spec)
        spectral_tensor = self.spectral_transform(spec_img)

        item = {
            "spatial": spatial_tensor,
            "spectral": spectral_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": img_path,
            "is_corrupt": is_corrupt,
        }
        if self.source_groups is not None:
            item["source_group"] = self.source_groups[idx]
        return item


def create_dataset_splits_from_manifest(
    manifest_path: str,
    img_size: int = 256,
) -> Tuple[DualDomainDataset, DualDomainDataset, DualDomainDataset]:
    """Loads strictly group-disjoint train/val/test datasets from a manifest JSON."""
    payload = json.loads(Path(manifest_path).read_text())
    if payload.get("status") != "valid_group_disjoint_split":
        raise ValueError(
            f"Manifest {manifest_path} does not contain a valid group-disjoint split: "
            f"{payload.get('error', payload.get('status'))}"
        )

    splits = payload["splits"]
    train_items = splits["train"]
    val_items = splits["val"]
    test_items = splits["test"]

    train_ds = DualDomainDataset(
        [x["path"] for x in train_items],
        [x["label"] for x in train_items],
        is_train=True,
        img_size=img_size,
        source_groups=[x["source_group"] for x in train_items],
    )
    val_ds = DualDomainDataset(
        [x["path"] for x in val_items],
        [x["label"] for x in val_items],
        is_train=False,
        img_size=img_size,
        source_groups=[x["source_group"] for x in val_items],
    )
    test_ds = DualDomainDataset(
        [x["path"] for x in test_items],
        [x["label"] for x in test_items],
        is_train=False,
        img_size=img_size,
        source_groups=[x["source_group"] for x in test_items],
    )
    return train_ds, val_ds, test_ds


def create_dataset_splits_from_dir(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_group_split: bool = True,
    manifest_path: Optional[str] = None,
    near_dup_threshold: int = 4,
    img_size: int = 256,
) -> Tuple[DualDomainDataset, DualDomainDataset, DualDomainDataset]:
    """
    Scans a directory and builds train/val/test DualDomainDataset splits.
    Supports:
    - Manifest mode (`manifest_path`): loads group-disjoint splits generated by StratifiedGroupKFold.
    - Format 1 (Pre-split): `data_dir/train/{real,fake}`, `val/{real,fake}`, and `test/{real,fake}`.
    - Format 2 (Unified): `data_dir/real` and `data_dir/fake`, split by source-group provenance
      and perceptual clustering via StratifiedGroupKFold when `use_group_split=True`.
    """
    if manifest_path is not None and Path(manifest_path).exists():
        return create_dataset_splits_from_manifest(manifest_path, img_size=img_size)

    data_path = Path(data_dir)
    exts = IMAGE_EXTENSIONS

    if (data_path / "train").exists() and (data_path / "val").exists():
        train_real = sorted([str(p) for p in (data_path / "train" / "real").glob("*.*") if p.suffix.lower() in exts])
        train_fake = sorted([str(p) for p in (data_path / "train" / "fake").glob("*.*") if p.suffix.lower() in exts])
        train_paths = train_real + train_fake
        train_labels = [0] * len(train_real) + [1] * len(train_fake)

        val_real = sorted([str(p) for p in (data_path / "val" / "real").glob("*.*") if p.suffix.lower() in exts])
        val_fake = sorted([str(p) for p in (data_path / "val" / "fake").glob("*.*") if p.suffix.lower() in exts])
        val_paths = val_real + val_fake
        val_labels = [0] * len(val_real) + [1] * len(val_fake)

        test_dir = data_path / "test"
        is_val_fallback = False
        if test_dir.exists():
            test_real = sorted([str(p) for p in (test_dir / "real").glob("*.*") if p.suffix.lower() in exts]) if (test_dir / "real").exists() else []
            test_fake = sorted([str(p) for p in (test_dir / "fake").glob("*.*") if p.suffix.lower() in exts]) if (test_dir / "fake").exists() else []
            test_paths = test_real + test_fake
            test_labels = [0] * len(test_real) + [1] * len(test_fake)
            if not test_paths:
                raise ValueError(
                    f"Test split directory exists at {test_dir} but contains 0 valid images."
                )
        else:
            logger.warning("No test directory at %s; falling back to validation split.", test_dir)
            test_paths, test_labels = val_paths, val_labels
            is_val_fallback = True

        train_ds = DualDomainDataset(train_paths, train_labels, is_train=True, img_size=img_size)
        val_ds = DualDomainDataset(val_paths, val_labels, is_train=False, img_size=img_size)
        test_ds = DualDomainDataset(test_paths, test_labels, is_train=False, img_size=img_size)
        test_ds.is_val_fallback = is_val_fallback
        return train_ds, val_ds, test_ds

    # Format 2: Unified real / fake folder
    if use_group_split:
        scan = scan_dataset_images(data_path)
        records = scan["records"]
        if not records:
            raise ValueError(f"No valid images found under {data_path / 'real'} or {data_path / 'fake'}.")
        detect_near_duplicates(records, phash_threshold=near_dup_threshold)
        assign_source_groups(records, data_path)

        indices = list(range(len(records)))
        labels = [r["label"] for r in records]
        groups = [r["source_group"] for r in records]
        test_ratio = max(0.05, 1.0 - train_ratio - val_ratio)
        train_i, val_i, test_i = stratified_group_train_val_test_split(
            indices, labels, groups, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        train_ds = DualDomainDataset(
            [records[i]["path"] for i in train_i],
            [records[i]["label"] for i in train_i],
            is_train=True,
            img_size=img_size,
            source_groups=[records[i]["source_group"] for i in train_i],
        )
        val_ds = DualDomainDataset(
            [records[i]["path"] for i in val_i],
            [records[i]["label"] for i in val_i],
            is_train=False,
            img_size=img_size,
            source_groups=[records[i]["source_group"] for i in val_i],
        )
        test_ds = DualDomainDataset(
            [records[i]["path"] for i in test_i],
            [records[i]["label"] for i in test_i],
            is_train=False,
            img_size=img_size,
            source_groups=[records[i]["source_group"] for i in test_i],
        )
        return train_ds, val_ds, test_ds

    # Explicit opt-out random split fallback
    real_paths = sorted([str(p) for p in (data_path / "real").glob("*.*") if p.suffix.lower() in exts])
    fake_paths = sorted([str(p) for p in (data_path / "fake").glob("*.*") if p.suffix.lower() in exts])

    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_paths))
    rng.shuffle(indices)

    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    n_total = len(all_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_paths, train_labels = all_paths[:n_train], all_labels[:n_train]
    val_paths, val_labels = all_paths[n_train:n_train + n_val], all_labels[n_train:n_train + n_val]
    test_paths, test_labels = all_paths[n_train + n_val:], all_labels[n_train + n_val:]

    train_ds = DualDomainDataset(train_paths, train_labels, is_train=True, img_size=img_size)
    val_ds = DualDomainDataset(val_paths, val_labels, is_train=False, img_size=img_size)
    test_ds = DualDomainDataset(test_paths, test_labels, is_train=False, img_size=img_size)
    return train_ds, val_ds, test_ds
