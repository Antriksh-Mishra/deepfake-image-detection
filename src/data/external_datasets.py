"""
External Dataset Loaders & Standardized Preprocessing Interfaces.

Supports independently sourced benchmark datasets:
1. FaceForensics++ (FF++): original_sequences vs manipulated_sequences (c23/c40/raw, frames or videos).
2. Celeb-DF (v1/v2): Celeb-real, YouTube-real vs Celeb-synthesis (+ optional List_of_testing_videos.txt).
3. DFDC: metadata.json / labels.csv video or extracted-frame layouts.

Enforces uniform label mapping (0 = Authentic Real, 1 = Manipulated Fake),
never trains on external samples, and handles unavailable datasets gracefully.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class ExternalSample:
    """Standardized sample record for external benchmark evaluation."""
    path: str
    label: int  # 0 = Authentic Real, 1 = Manipulated Fake
    dataset_name: str
    source_id: str
    sub_category: str = "default"
    media_type: str = "image"  # 'image' or 'video'
    frame_index: Optional[int] = None


@dataclass
class ExternalDatasetInfo:
    """Availability & metadata summary for an external dataset."""
    key: str
    display_name: str
    root_dir: str
    available: bool
    status: str  # 'available', 'unavailable', 'empty'
    reason: str
    sample_count: int = 0
    real_count: int = 0
    fake_count: int = 0
    sub_categories: Dict[str, int] = field(default_factory=dict)
    samples: List[ExternalSample] = field(default_factory=list)


def resolve_external_dataset_roots(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Resolves dataset root paths from environment variables, config YAML, or standard `data/external/` locations.
    """
    cfg_ext = (config or {}).get("external_datasets", {})
    root_base = Path(cfg_ext.get("root_dir", "data/external"))

    return {
        "faceforensics": Path(os.environ.get("FFPP_DIR") or cfg_ext.get("faceforensics_dir", root_base / "FaceForensics++")),
        "celeb_df": Path(os.environ.get("CELEBDF_DIR") or cfg_ext.get("celeb_df_dir", root_base / "Celeb-DF")),
        "dfdc": Path(os.environ.get("DFDC_DIR") or cfg_ext.get("dfdc_dir", root_base / "DFDC")),
    }


def extract_uniform_video_frames(
    video_path: Path,
    max_frames: int = 8,
) -> List[Tuple[int, Image.Image]]:
    """
    Extracts up to `max_frames` uniformly spaced RGB frames from a video file using OpenCV.
    Returns list of (frame_index, PIL.Image).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video container: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        extracted: List[Tuple[int, Image.Image]] = []

        if total_frames > 0:
            n_sample = min(max_frames, total_frames)
            target_indices = np.linspace(0, total_frames - 1, num=n_sample, dtype=int).tolist()
            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    extracted.append((int(idx), Image.fromarray(rgb)))
        else:
            # Fallback sequential read if container omits frame count metadata
            frame_idx = 0
            while len(extracted) < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx % 5 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    extracted.append((frame_idx, Image.fromarray(rgb)))
                frame_idx += 1

        return extracted
    finally:
        cap.release()


def _collect_media_files(directory: Path, compression: Optional[str] = None) -> List[Path]:
    """Recursively collects supported image or video files, optionally filtering compression path segment."""
    if not directory.exists():
        return []
    files: List[Path] = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMAGE_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
            continue
        if compression and compression.lower() != "all":
            parts_lower = [part.lower() for part in p.parts]
            # Filter only if standard FF++ compression tier folder exists in hierarchy
            has_tier_folder = any(t in parts_lower for t in ("c23", "c40", "raw"))
            if has_tier_folder and compression.lower() not in parts_lower:
                continue
        files.append(p)
    return files


def load_faceforensics_samples(
    root_dir: Path,
    compression: str = "all",
    max_per_class: Optional[int] = None,
) -> ExternalDatasetInfo:
    """
    Loader for FaceForensics++ (FF++).

    Supported directory layouts:
    - Standard FF++: `original_sequences/` (Real -> 0) and `manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures,FaceShifter}/` (Fake -> 1)
    - Pre-extracted flat split: `real/` (0) and `fake/` (1)
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        return ExternalDatasetInfo(
            key="faceforensics",
            display_name="FaceForensics++",
            root_dir=str(root_path),
            available=False,
            status="unavailable",
            reason=(
                f"Directory not found at '{root_path}'. Download FaceForensics++ with official "
                "approval and place `original_sequences/` & `manipulated_sequences/` (or `real/` & `fake/`) here."
            ),
        )

    real_samples: List[ExternalSample] = []
    fake_samples: List[ExternalSample] = []
    sub_counts: Dict[str, int] = {}

    orig_dir = root_path / "original_sequences"
    manip_dir = root_path / "manipulated_sequences"

    if orig_dir.exists() or manip_dir.exists():
        if orig_dir.exists():
            for p in _collect_media_files(orig_dir, compression=compression):
                media_t = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "image"
                real_samples.append(
                    ExternalSample(
                        path=str(p),
                        label=0,
                        dataset_name="FaceForensics++",
                        source_id=f"ffpp_real_{p.stem.split('_')[0]}",
                        sub_category="original",
                        media_type=media_t,
                    )
                )
            if real_samples:
                sub_counts["original"] = len(real_samples)

        if manip_dir.exists():
            for p in _collect_media_files(manip_dir, compression=compression):
                media_t = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "image"
                rel_parts = p.relative_to(manip_dir).parts
                method_name = rel_parts[0] if rel_parts else "manipulated"
                src_vid = p.stem.split("_")[0]
                fake_samples.append(
                    ExternalSample(
                        path=str(p),
                        label=1,
                        dataset_name="FaceForensics++",
                        source_id=f"ffpp_fake_{src_vid}",
                        sub_category=method_name,
                        media_type=media_t,
                    )
                )
                sub_counts[method_name] = sub_counts.get(method_name, 0) + 1
    else:
        # Simplified real/fake layout
        for p in _collect_media_files(root_path / "real"):
            media_t = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "image"
            real_samples.append(
                ExternalSample(
                    path=str(p),
                    label=0,
                    dataset_name="FaceForensics++",
                    source_id=f"ffpp_real_{p.stem}",
                    sub_category="original",
                    media_type=media_t,
                )
            )
        for p in _collect_media_files(root_path / "fake"):
            media_t = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "image"
            fake_samples.append(
                ExternalSample(
                    path=str(p),
                    label=1,
                    dataset_name="FaceForensics++",
                    source_id=f"ffpp_fake_{p.stem.split('_')[0]}",
                    sub_category="manipulated",
                    media_type=media_t,
                )
            )
        if real_samples:
            sub_counts["original"] = len(real_samples)
        if fake_samples:
            sub_counts["manipulated"] = len(fake_samples)

    if max_per_class is not None:
        real_samples = real_samples[:max_per_class]
        fake_samples = fake_samples[:max_per_class]

    all_samples = real_samples + fake_samples
    if not all_samples:
        return ExternalDatasetInfo(
            key="faceforensics",
            display_name="FaceForensics++",
            root_dir=str(root_path),
            available=False,
            status="empty",
            reason=f"Directory '{root_path}' exists but contains no supported FF++ image or video files.",
        )

    return ExternalDatasetInfo(
        key="faceforensics",
        display_name="FaceForensics++",
        root_dir=str(root_path),
        available=True,
        status="available",
        reason="Loaded FaceForensics++ external samples cleanly.",
        sample_count=len(all_samples),
        real_count=len(real_samples),
        fake_count=len(fake_samples),
        sub_categories=sub_counts,
        samples=all_samples,
    )


def load_celebdf_samples(
    root_dir: Path,
    use_official_test_list: bool = True,
    max_per_class: Optional[int] = None,
) -> ExternalDatasetInfo:
    """
    Loader for Celeb-DF (v1 / v2).

    Note on official Celeb-DF `List_of_testing_videos.txt` convention:
    In Celeb-DF's text file, `1` = Real and `0` = Fake. This loader normalizes
    strictly to project convention: **0 = Authentic Real, 1 = Manipulated Fake**.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        return ExternalDatasetInfo(
            key="celeb_df",
            display_name="Celeb-DF",
            root_dir=str(root_path),
            available=False,
            status="unavailable",
            reason=(
                f"Directory not found at '{root_path}'. Request Celeb-DF access and place "
                "`Celeb-real/`, `YouTube-real/`, `Celeb-synthesis/`, and optional `List_of_testing_videos.txt` here."
            ),
        )

    real_samples: List[ExternalSample] = []
    fake_samples: List[ExternalSample] = []
    sub_counts: Dict[str, int] = {}

    test_list_file = root_path / "List_of_testing_videos.txt"
    if use_official_test_list and test_list_file.exists():
        for line in test_list_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            raw_lbl_str, rel_video = parts
            vid_path = root_path / rel_video.strip()
            if not vid_path.exists():
                continue
            # Celeb-DF convention: 1 -> Real (our label 0), 0 -> Fake (our label 1)
            mapped_label = 0 if raw_lbl_str.strip() == "1" else 1
            media_t = "video" if vid_path.suffix.lower() in VIDEO_EXTENSIONS else "image"
            sub_cat = Path(rel_video).parts[0] if Path(rel_video).parts else ("real" if mapped_label == 0 else "synthesis")
            sample = ExternalSample(
                path=str(vid_path),
                label=mapped_label,
                dataset_name="Celeb-DF",
                source_id=f"celebdf_{vid_path.stem.split('_')[0]}",
                sub_category=sub_cat,
                media_type=media_t,
            )
            if mapped_label == 0:
                real_samples.append(sample)
            else:
                fake_samples.append(sample)
            sub_counts[sub_cat] = sub_counts.get(sub_cat, 0) + 1
    else:
        for folder_name, mapped_label in (
            ("Celeb-real", 0),
            ("YouTube-real", 0),
            ("real", 0),
            ("Celeb-synthesis", 1),
            ("fake", 1),
        ):
            folder = root_path / folder_name
            if not folder.exists():
                continue
            for p in _collect_media_files(folder):
                media_t = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "image"
                sample = ExternalSample(
                    path=str(p),
                    label=mapped_label,
                    dataset_name="Celeb-DF",
                    source_id=f"celebdf_{p.stem.split('_')[0]}",
                    sub_category=folder_name,
                    media_type=media_t,
                )
                if mapped_label == 0:
                    real_samples.append(sample)
                else:
                    fake_samples.append(sample)
                sub_counts[folder_name] = sub_counts.get(folder_name, 0) + 1

    if max_per_class is not None:
        real_samples = real_samples[:max_per_class]
        fake_samples = fake_samples[:max_per_class]

    all_samples = real_samples + fake_samples
    if not all_samples:
        return ExternalDatasetInfo(
            key="celeb_df",
            display_name="Celeb-DF",
            root_dir=str(root_path),
            available=False,
            status="empty",
            reason=f"Directory '{root_path}' exists but contains no supported Celeb-DF images/videos.",
        )

    return ExternalDatasetInfo(
        key="celeb_df",
        display_name="Celeb-DF",
        root_dir=str(root_path),
        available=True,
        status="available",
        reason="Loaded Celeb-DF external samples cleanly.",
        sample_count=len(all_samples),
        real_count=len(real_samples),
        fake_count=len(fake_samples),
        sub_categories=sub_counts,
        samples=all_samples,
    )


def _normalize_dfdc_label(raw_val: Any) -> Optional[int]:
    if raw_val is None:
        return None
    s = str(raw_val).strip().upper()
    if s in {"REAL", "0", "FALSE", "AUTHENTIC"}:
        return 0
    if s in {"FAKE", "1", "TRUE", "MANIPULATED"}:
        return 1
    return None


def load_dfdc_samples(
    root_dir: Path,
    max_per_class: Optional[int] = None,
) -> ExternalDatasetInfo:
    """
    Loader for Deepfake Detection Challenge (DFDC) dataset.

    Supports:
    - Standard DFDC `metadata.json` (`{"clip.mp4": {"label": "REAL"|"FAKE", "original": ...}}`)
    - Public test CSV `labels.csv` (`filename,label`)
    - Simplified `real/` and `fake/` extracted frame folders
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        return ExternalDatasetInfo(
            key="dfdc",
            display_name="DFDC",
            root_dir=str(root_path),
            available=False,
            status="unavailable",
            reason=(
                f"Directory not found at '{root_path}'. Download accessible DFDC archive and place "
                "clips with `metadata.json` / `labels.csv` (or `real/` & `fake/`) here."
            ),
        )

    real_samples: List[ExternalSample] = []
    fake_samples: List[ExternalSample] = []
    seen_paths: set = set()
    sub_counts: Dict[str, int] = {}

    # 1. Check metadata.json files (supports nested dfdc_train_part_* folders)
    for meta_file in sorted(root_path.rglob("metadata.json")):
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        base_folder = meta_file.parent
        for fname, info in meta.items():
            if not isinstance(info, dict):
                continue
            lbl = _normalize_dfdc_label(info.get("label"))
            if lbl is None:
                continue
            cand = base_folder / fname
            if not cand.exists():
                # Check if extracted frames directory exists with matching stem
                stem_dir = base_folder / Path(fname).stem
                if stem_dir.is_dir():
                    frame_files = _collect_media_files(stem_dir)
                    for fp in frame_files:
                        sp = str(fp)
                        if sp in seen_paths:
                            continue
                        seen_paths.add(sp)
                        orig_id = Path(str(info.get("original") or fname)).stem
                        sample = ExternalSample(
                            path=sp,
                            label=lbl,
                            dataset_name="DFDC",
                            source_id=f"dfdc_{orig_id}",
                            sub_category="metadata_frames",
                            media_type="image",
                        )
                        (real_samples if lbl == 0 else fake_samples).append(sample)
                        sub_counts["metadata_frames"] = sub_counts.get("metadata_frames", 0) + 1
                continue
            sp = str(cand)
            if sp in seen_paths:
                continue
            seen_paths.add(sp)
            media_t = "video" if cand.suffix.lower() in VIDEO_EXTENSIONS else "image"
            orig_id = Path(str(info.get("original") or fname)).stem
            sample = ExternalSample(
                path=sp,
                label=lbl,
                dataset_name="DFDC",
                source_id=f"dfdc_{orig_id}",
                sub_category="metadata_json",
                media_type=media_t,
            )
            (real_samples if lbl == 0 else fake_samples).append(sample)
            sub_counts["metadata_json"] = sub_counts.get("metadata_json", 0) + 1

    # 2. Check labels.csv files
    for csv_file in sorted(root_path.rglob("labels.csv")):
        try:
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename") or row.get("video") or row.get("file")
                    lbl = _normalize_dfdc_label(row.get("label") or row.get("is_fake"))
                    if not fname or lbl is None:
                        continue
                    cand = csv_file.parent / fname
                    if not cand.exists() and (csv_file.parent / "test_videos" / fname).exists():
                        cand = csv_file.parent / "test_videos" / fname
                    if not cand.exists():
                        continue
                    sp = str(cand)
                    if sp in seen_paths:
                        continue
                    seen_paths.add(sp)
                    media_t = "video" if cand.suffix.lower() in VIDEO_EXTENSIONS else "image"
                    sample = ExternalSample(
                        path=sp,
                        label=lbl,
                        dataset_name="DFDC",
                        source_id=f"dfdc_{cand.stem}",
                        sub_category="labels_csv",
                        media_type=media_t,
                    )
                    (real_samples if lbl == 0 else fake_samples).append(sample)
                    sub_counts["labels_csv"] = sub_counts.get("labels_csv", 0) + 1
        except Exception:
            continue

    # 3. Simplified real / fake directories fallback
    if not real_samples and not fake_samples:
        for folder_name, lbl in (("real", 0), ("fake", 1)):
            for p in _collect_media_files(root_path / folder_name):
                media_t = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "image"
                sample = ExternalSample(
                    path=str(p),
                    label=lbl,
                    dataset_name="DFDC",
                    source_id=f"dfdc_{p.stem}",
                    sub_category=folder_name,
                    media_type=media_t,
                )
                (real_samples if lbl == 0 else fake_samples).append(sample)
                sub_counts[folder_name] = sub_counts.get(folder_name, 0) + 1

    if max_per_class is not None:
        real_samples = real_samples[:max_per_class]
        fake_samples = fake_samples[:max_per_class]

    all_samples = real_samples + fake_samples
    if not all_samples:
        return ExternalDatasetInfo(
            key="dfdc",
            display_name="DFDC",
            root_dir=str(root_path),
            available=False,
            status="empty",
            reason=f"Directory '{root_path}' exists but contains no readable DFDC metadata/media pairs.",
        )

    return ExternalDatasetInfo(
        key="dfdc",
        display_name="DFDC",
        root_dir=str(root_path),
        available=True,
        status="available",
        reason="Loaded DFDC external samples cleanly.",
        sample_count=len(all_samples),
        real_count=len(real_samples),
        fake_count=len(fake_samples),
        sub_categories=sub_counts,
        samples=all_samples,
    )


def discover_external_datasets(
    config: Optional[Dict[str, Any]] = None,
    compression: str = "all",
    max_per_class: Optional[int] = None,
) -> Dict[str, ExternalDatasetInfo]:
    """Discovers and inspects availability for FaceForensics++, Celeb-DF, and DFDC."""
    roots = resolve_external_dataset_roots(config)
    return {
        "faceforensics": load_faceforensics_samples(
            roots["faceforensics"], compression=compression, max_per_class=max_per_class
        ),
        "celeb_df": load_celebdf_samples(
            roots["celeb_df"], max_per_class=max_per_class
        ),
        "dfdc": load_dfdc_samples(
            roots["dfdc"], max_per_class=max_per_class
        ),
    }
