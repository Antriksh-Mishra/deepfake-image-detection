"""
Dataset Integrity, Duplicate Detection, Source-Group Clustering, and Leakage Audit Module.

Provides:
1. SHA-256 exact byte duplicate & cross-split / cross-label conflict detection.
2. Perceptual hashing (64-bit 2D DCT pHash) & near-duplicate detection with chunked
   vectorized Hamming distance computation and bounded star/leader clustering.
3. Provenance-aware source-group resolution and leakage-free splitting via
   sklearn.model_selection.StratifiedGroupKFold.
4. Corrupted image verification, resolution profiling, and structured JSON reporting
   to results/dataset_audit.json without mutating or deleting original files.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dct
from sklearn.model_selection import StratifiedGroupKFold

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 digest of raw file bytes on disk."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _bits_to_hex(bits: np.ndarray) -> str:
    """Converts a 64-element binary vector to a 16-char hexadecimal string."""
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def compute_phash(
    image: Image.Image,
    hash_size: int = 8,
    highfreq_factor: int = 4,
    exclude_dc: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    Computes 64-bit 2D DCT perceptual hash (pHash).

    Returns:
        (phash_hex, phash_bits): 16-char hex digest and shape-(64,) uint8 array.
    """
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)

    dct_coeff = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low_freq = dct_coeff[:hash_size, :hash_size]
    flat = low_freq.flatten()
    ref_vals = flat[1:] if (exclude_dc and len(flat) > 1) else flat
    med = float(np.median(ref_vals))

    bits = (flat > med).astype(np.uint8)
    return _bits_to_hex(bits), bits


def phash_hamming_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    """Computes Hamming distance between two binary pHash vectors."""
    return int(np.sum(np.asarray(bits_a, dtype=np.uint8) != np.asarray(bits_b, dtype=np.uint8)))


def _infer_structured_source_id(file_path: Path, data_dir: Path) -> Optional[str]:
    """
    Extracts source group ID from standard forensic dataset naming conventions
    (FaceForensics++, Celeb-DF, DFDC) or explicit metadata naming when present.
    Returns None for unlabeled flat portraits so perceptual clustering handles them.
    """
    stem = file_path.stem
    parts = [p.lower() for p in file_path.relative_to(data_dir).parts]
    joined = "/".join(parts)

    # Celeb-DF naming convention: id<src>_id<target>_<clip> or id<src>_<clip>
    celeb_match = re.match(r"^(id\d+)_", stem, re.IGNORECASE)
    if celeb_match and ("celeb" in joined or "synthesis" in joined or "youtube" in joined):
        return f"celebdf_{celeb_match.group(1).lower()}"

    # FaceForensics++ pair convention: <src>_<tgt>_frame<n>
    ff_match = re.match(r"^(\d{3})_\d{3}(?:_|$)", stem)
    if ff_match:
        return f"ffpp_src_{ff_match.group(1)}"

    return None


def scan_dataset_images(
    data_dir: Path,
    manifest_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Non-destructively scans all images under data_dir/{train,val,test}/{real,fake}
    or data_dir/{real,fake}, verifying file readability, resolutions, SHA-256, and pHash.
    Also checks any optional manifest_paths for missing files.
    """
    records: List[Dict[str, Any]] = []
    corrupted_files: List[Dict[str, str]] = []
    missing_files: List[str] = []
    resolution_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    split_counts: Counter = Counter()
    class_counts: Counter = Counter()
    split_class_counts: Dict[str, Counter] = defaultdict(Counter)

    has_presplit = (data_dir / "train").exists() and (data_dir / "val").exists()
    candidate_entries: List[Tuple[str, str, int, Path]] = []

    if has_presplit:
        for split in ("train", "val", "test"):
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            for cls_name, lbl in (("real", 0), ("fake", 1)):
                cls_dir = split_dir / cls_name
                if not cls_dir.exists():
                    continue
                for fpath in sorted(cls_dir.iterdir()):
                    if fpath.is_symlink() and not fpath.exists():
                        missing_files.append(str(fpath))
                        continue
                    if fpath.is_file() and fpath.suffix.lower() in IMAGE_EXTENSIONS:
                        candidate_entries.append((split, cls_name, lbl, fpath))
    else:
        for cls_name, lbl in (("real", 0), ("fake", 1)):
            cls_dir = data_dir / cls_name
            if not cls_dir.exists():
                continue
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.is_symlink() and not fpath.exists():
                    missing_files.append(str(fpath))
                    continue
                if fpath.is_file() and fpath.suffix.lower() in IMAGE_EXTENSIONS:
                    candidate_entries.append(("unsplit", cls_name, lbl, fpath))

    # Check explicit expected manifest paths for missing files if supplied
    if manifest_paths:
        for rel_or_abs in manifest_paths:
            p = Path(rel_or_abs)
            if not p.is_absolute():
                p = data_dir / p
            if not p.exists():
                missing_files.append(str(p))

    for idx, (split, cls_name, lbl, fpath) in enumerate(candidate_entries):
        split_counts[split] += 1
        class_counts[cls_name] += 1
        split_class_counts[split][cls_name] += 1

        try:
            file_size = fpath.stat().st_size
            sha256 = compute_file_sha256(fpath)

            # Verify structural integrity then decode RGB
            with Image.open(fpath) as verify_im:
                verify_im.verify()

            with Image.open(fpath) as im:
                im = ImageOps.exif_transpose(im)
                mode_counts[im.mode] += 1
                width, height = im.size
                resolution_counts[f"{width}x{height}"] += 1
                rgb = im.convert("RGB")
                phash_hex, phash_bits = compute_phash(rgb)
                flipped = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                _, phash_flip_bits = compute_phash(flipped)

            records.append(
                {
                    "index": len(records),
                    "path": str(fpath),
                    "rel_path": str(fpath.relative_to(data_dir)),
                    "filename": fpath.name,
                    "split": split,
                    "class_name": cls_name,
                    "label": int(lbl),
                    "file_size_bytes": int(file_size),
                    "width": int(width),
                    "height": int(height),
                    "sha256": sha256,
                    "phash": phash_hex,
                    "phash_bits": phash_bits,
                    "phash_flip_bits": phash_flip_bits,
                }
            )
        except Exception as exc:
            corrupted_files.append(
                {
                    "path": str(fpath),
                    "split": split,
                    "class_name": cls_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "has_presplit": has_presplit,
        "records": records,
        "corrupted_files": corrupted_files,
        "missing_files": sorted(set(missing_files)),
        "resolution_distribution": dict(sorted(resolution_counts.items())),
        "mode_distribution": dict(sorted(mode_counts.items())),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "split_class_distribution": {k: dict(v) for k, v in split_class_counts.items()},
    }


def detect_exact_duplicates(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detects exact SHA-256 duplicate groups, redundant counts, and split/label leaks."""
    by_sha: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_sha[rec["sha256"]].append(rec)

    dup_groups = []
    cross_split_groups = 0
    cross_label_groups = 0
    redundant_copies = 0
    total_files_in_dup_groups = 0

    for sha, items in by_sha.items():
        if len(items) <= 1:
            continue
        splits_involved = sorted({it["split"] for it in items})
        labels_involved = sorted({it["class_name"] for it in items})
        is_cross_split = len(splits_involved) > 1
        is_cross_label = len(labels_involved) > 1

        if is_cross_split:
            cross_split_groups += 1
        if is_cross_label:
            cross_label_groups += 1

        redundant_copies += len(items) - 1
        total_files_in_dup_groups += len(items)

        dup_groups.append(
            {
                "sha256": sha,
                "count": len(items),
                "splits": splits_involved,
                "labels": labels_involved,
                "cross_split_leakage": is_cross_split,
                "cross_label_conflict": is_cross_label,
                "files": [
                    {
                        "rel_path": it["rel_path"],
                        "split": it["split"],
                        "class_name": it["class_name"],
                    }
                    for it in items
                ],
            }
        )

    dup_groups.sort(key=lambda g: g["count"], reverse=True)
    return {
        "unique_sha256_count": len(by_sha),
        "duplicate_group_count": len(dup_groups),
        "duplicate_extra_copy_count": int(redundant_copies),
        "total_files_in_duplicate_groups": int(total_files_in_dup_groups),
        "cross_split_duplicate_group_count": int(cross_split_groups),
        "cross_label_conflict_group_count": int(cross_label_groups),
        "duplicate_groups": dup_groups,
    }


def detect_near_duplicates(
    records: List[Dict[str, Any]],
    phash_threshold: int = 4,
    block_size: int = 512,
    max_sample_pairs: int = 30,
) -> Dict[str, Any]:
    """
    Detects perceptual exact-pHash matches (dist == 0) and near-duplicates (dist <= threshold)
    using memory-bounded chunked vectorized Hamming computation and diameter-bounded
    leader clustering (avoiding transitive chaining across unrelated portraits).
    """
    n = len(records)
    if n == 0:
        return {
            "phash_threshold": phash_threshold,
            "unique_phash_count": 0,
            "exact_phash_collision_images": 0,
            "near_duplicate_image_count": 0,
            "cross_split_near_duplicate_image_count": 0,
            "near_duplicate_cluster_count": 0,
            "multi_member_cluster_count": 0,
            "similarity_summary": {},
            "sample_cross_split_near_duplicates": [],
        }

    bits = np.stack([r["phash_bits"] for r in records]).astype(np.int16)
    flip_bits = np.stack([r["phash_flip_bits"] for r in records]).astype(np.int16)
    row_sums = bits.sum(axis=1, keepdims=True)
    flip_sums = flip_bits.sum(axis=1, keepdims=True)

    has_exact_phash_match = np.zeros(n, dtype=bool)
    has_near_dup = np.zeros(n, dtype=bool)
    has_cross_split_near_dup = np.zeros(n, dtype=bool)
    min_neighbor_dists = np.full(n, 64, dtype=np.int16)
    min_cross_split_dists = np.full(n, 64, dtype=np.int16)

    splits_arr = np.array([r["split"] for r in records])
    shas_arr = [r["sha256"] for r in records]

    # Sample interesting cross-split near-duplicates (non-identical SHA-256)
    sample_pairs: List[Dict[str, Any]] = []

    # Leader/star clustering assignments (leader index -> members)
    assigned_leader = np.full(n, -1, dtype=np.int32)
    leaders: List[int] = []

    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        blk = bits[start:end]
        blk_sum = row_sums[start:end]

        d_orig = blk_sum + row_sums.T - 2 * (blk @ bits.T)
        d_flip = blk_sum + flip_sums.T - 2 * (blk @ flip_bits.T)
        d_min = np.minimum(d_orig, d_flip)

        for local_i in range(end - start):
            global_i = start + local_i
            row = d_min[local_i]
            row[global_i] = 999  # Exclude self

            min_d = int(row.min())
            min_neighbor_dists[global_i] = min_d
            if min_d == 0:
                has_exact_phash_match[global_i] = True
            if min_d <= phash_threshold:
                has_near_dup[global_i] = True

            cross_mask = splits_arr != splits_arr[global_i]
            if np.any(cross_mask):
                cross_dists = row[cross_mask]
                min_cross = int(cross_dists.min())
                min_cross_split_dists[global_i] = min_cross
                if min_cross <= phash_threshold:
                    has_cross_split_near_dup[global_i] = True

                    # Record representative non-SHA-identical cross-split pair
                    if len(sample_pairs) < max_sample_pairs:
                        cross_indices = np.where((row <= phash_threshold) & cross_mask)[0]
                        for j in cross_indices:
                            j_int = int(j)
                            if shas_arr[global_i] != shas_arr[j_int] and global_i < j_int:
                                dist_val = int(row[j_int])
                                sample_pairs.append(
                                    {
                                        "file_a": records[global_i]["rel_path"],
                                        "split_a": records[global_i]["split"],
                                        "class_a": records[global_i]["class_name"],
                                        "file_b": records[j_int]["rel_path"],
                                        "split_b": records[j_int]["split"],
                                        "class_b": records[j_int]["class_name"],
                                        "hamming_distance": dist_val,
                                        "similarity_score": round(1.0 - (dist_val / 64.0), 4),
                                    }
                                )
                                break

            # Bounded greedy leader assignment (prevents transitive chaining)
            if assigned_leader[global_i] == -1:
                best_leader = -1
                best_leader_dist = phash_threshold + 1
                if leaders:
                    leader_arr = np.array(leaders, dtype=np.int32)
                    l_dists = row[leader_arr]
                    min_idx = int(np.argmin(l_dists))
                    if int(l_dists[min_idx]) <= phash_threshold:
                        best_leader = int(leader_arr[min_idx])
                        best_leader_dist = int(l_dists[min_idx])

                if best_leader != -1 and best_leader_dist <= phash_threshold:
                    assigned_leader[global_i] = best_leader
                else:
                    leaders.append(global_i)
                    assigned_leader[global_i] = global_i

    cluster_sizes = Counter(int(x) for x in assigned_leader)
    multi_clusters = sum(1 for sz in cluster_sizes.values() if sz > 1)
    unique_phashes = len({r["phash"] for r in records})

    for idx, rec in enumerate(records):
        rec["phash_cluster_id"] = f"phash_cluster_{int(assigned_leader[idx]):04d}"

    return {
        "phash_threshold": int(phash_threshold),
        "unique_phash_count": int(unique_phashes),
        "exact_phash_collision_images": int(np.sum(has_exact_phash_match)),
        "near_duplicate_image_count": int(np.sum(has_near_dup)),
        "cross_split_near_duplicate_image_count": int(np.sum(has_cross_split_near_dup)),
        "near_duplicate_cluster_count": int(len(cluster_sizes)),
        "multi_member_cluster_count": int(multi_clusters),
        "similarity_summary": {
            "nearest_neighbor_hamming_min": int(np.min(min_neighbor_dists)),
            "nearest_neighbor_hamming_median": float(np.median(min_neighbor_dists)),
            "nearest_neighbor_hamming_p90": float(np.percentile(min_neighbor_dists, 90)),
            "nearest_neighbor_hamming_max": int(np.max(min_neighbor_dists)),
            "nearest_cross_split_hamming_median": float(np.median(min_cross_split_dists)),
            "mean_nearest_similarity": round(float(np.mean(1.0 - min_neighbor_dists / 64.0)), 4),
        },
        "sample_cross_split_near_duplicates": sorted(
            sample_pairs, key=lambda x: x["hamming_distance"]
        )[:max_sample_pairs],
    }


def _load_anchor_hashes(data_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Loads base authentic portrait pHashes from data_dir/demo_samples (and root
    whatsapp_test_input.jpg if present) to recognize portrait identity derivations.
    """
    anchors: List[Tuple[str, np.ndarray, np.ndarray]] = []
    anchor_candidates: List[Path] = []

    demo_dir = data_dir / "demo_samples"
    if demo_dir.exists():
        # Prefer authentic base portraits first to avoid fake-variant collisions
        anchor_candidates.extend(sorted(demo_dir.glob("authentic_*.*")))
        if not anchor_candidates:
            anchor_candidates.extend(sorted(demo_dir.glob("*.*")))

    whatsapp_path = data_dir.parent / "whatsapp_test_input.jpg"
    if whatsapp_path.exists():
        anchor_candidates.append(whatsapp_path)

    for p in anchor_candidates:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(p) as im:
                    rgb = im.convert("RGB")
                    _, b_orig = compute_phash(rgb)
                    _, b_flip = compute_phash(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
                    anchors.append((p.stem, b_orig, b_flip))
            except Exception:
                continue
    return anchors


def assign_source_groups(
    records: List[Dict[str, Any]],
    data_dir: Path,
    anchor_max_hamming: int = 12,
) -> Dict[str, Any]:
    """
    Assigns every image record to a deterministic source identity group so all variations
    derived from the same original portrait share one group.

    Priority order:
    1. Explicit provenance manifest `<data_dir>/source_groups.json` if available.
    2. Structured dataset naming (FF++ / Celeb-DF source IDs).
    3. Exact SHA-256 equivalence + Reference anchor attribution + bounded pHash cluster.
    """
    manifest_file = data_dir / "source_groups.json"
    explicit_map: Dict[str, str] = {}
    if manifest_file.exists():
        try:
            loaded = json.loads(manifest_file.read_text())
            if isinstance(loaded, dict):
                explicit_map = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            explicit_map = {}

    anchors = _load_anchor_hashes(data_dir)

    # First ensure exact SHA-256 identical files ALWAYS map to the exact same source group
    sha_group: Dict[str, str] = {}
    provenance_method_counts: Counter = Counter()

    for rec in records:
        rel_path = rec["rel_path"]
        fname = rec["filename"]
        sha = rec["sha256"]

        if rel_path in explicit_map:
            grp = explicit_map[rel_path]
            method = "explicit_manifest"
        elif fname in explicit_map:
            grp = explicit_map[fname]
            method = "explicit_manifest"
        else:
            structured = _infer_structured_source_id(Path(rec["path"]), data_dir)
            if structured is not None:
                grp = structured
                method = "structured_filename_id"
            elif sha in sha_group:
                grp = sha_group[sha]
                method = "sha256_alias"
            elif anchors:
                bits = rec["phash_bits"]
                best_name = None
                best_dist = 999
                for a_name, a_orig, a_flip in anchors:
                    d = min(
                        phash_hamming_distance(bits, a_orig),
                        phash_hamming_distance(bits, a_flip),
                    )
                    if d < best_dist:
                        best_dist = d
                        best_name = a_name
                if best_name is not None and best_dist <= anchor_max_hamming:
                    grp = f"anchor_{best_name}"
                    method = "perceptual_anchor"
                else:
                    grp = rec.get("phash_cluster_id", f"file_{rec['index']:05d}")
                    method = "phash_leader_cluster"
            else:
                grp = rec.get("phash_cluster_id", f"file_{rec['index']:05d}")
                method = "phash_leader_cluster"

        # Enforce SHA-256 consistency across identically hashed files
        if sha in sha_group:
            grp = sha_group[sha]
        else:
            sha_group[sha] = grp

        rec["source_group"] = grp
        rec["group_method"] = method
        provenance_method_counts[method] += 1

    # Aggregate group distribution & cross-split leakage metrics
    group_counts: Counter = Counter(rec["source_group"] for rec in records)
    group_splits: Dict[str, set] = defaultdict(set)
    group_classes: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        g = rec["source_group"]
        group_splits[g].add(rec["split"])
        group_classes[g][rec["class_name"]] += 1

    cross_split_groups = sorted([g for g, sp in group_splits.items() if len(sp) > 1])
    group_details = {
        g: {
            "count": int(group_counts[g]),
            "splits": sorted(group_splits[g]),
            "class_counts": dict(group_classes[g]),
            "leaked_across_splits": len(group_splits[g]) > 1,
        }
        for g in sorted(group_counts.keys())
    }

    return {
        "total_source_groups": len(group_counts),
        "cross_split_leaked_group_count": len(cross_split_groups),
        "cross_split_leaked_groups": cross_split_groups,
        "group_assignment_methods": dict(provenance_method_counts),
        "source_group_distribution": group_details,
    }


def stratified_group_train_val_test_split(
    indices: Sequence[int],
    labels: Sequence[int],
    groups: Sequence[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits dataset indices into strictly group-disjoint (train, val, test) subsets
    using sklearn.model_selection.StratifiedGroupKFold.

    Raises:
        ValueError: If fewer than 3 distinct source groups exist or disjoint splitting fails.
    """
    n = len(indices)
    idx_arr = np.asarray(indices)
    y_arr = np.asarray(labels)
    g_arr = np.asarray(groups)

    unique_groups = np.unique(g_arr)
    n_groups = len(unique_groups)
    if n_groups < 3:
        raise ValueError(
            f"Cannot form group-disjoint train/val/test splits from only {n_groups} source group(s); "
            f"at least 3 distinct source groups are required."
        )

    # Step 1: Held-out Test Fold via StratifiedGroupKFold
    test_splits = max(3, min(n_groups, int(round(1.0 / max(test_ratio, 1e-3)))))
    sgkf_test = StratifiedGroupKFold(n_splits=test_splits, shuffle=True, random_state=seed)
    folds_outer = list(sgkf_test.split(idx_arr, y_arr, g_arr))

    # Choose test fold with closest sample fraction to test_ratio while having non-empty classes if possible
    target_test_n = max(1, int(round(n * test_ratio)))
    best_outer = min(
        folds_outer,
        key=lambda tr_te: (
            0 if len(np.unique(y_arr[tr_te[1]])) == len(np.unique(y_arr)) else 1,
            abs(len(tr_te[1]) - target_test_n),
        ),
    )
    train_val_pos, test_pos = best_outer

    # Step 2: Validation Fold from remaining Train+Val groups via StratifiedGroupKFold
    rem_idx = idx_arr[train_val_pos]
    rem_y = y_arr[train_val_pos]
    rem_g = g_arr[train_val_pos]
    rem_n_groups = len(np.unique(rem_g))
    if rem_n_groups < 2:
        raise ValueError(
            f"Only {rem_n_groups} group remained for train/val split after holding out test split."
        )

    val_frac_of_rem = val_ratio / max(1.0 - (len(test_pos) / max(n, 1)), 1e-3)
    val_splits = max(2, min(rem_n_groups, int(round(1.0 / max(val_frac_of_rem, 1e-3)))))
    sgkf_val = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed)
    folds_inner = list(sgkf_val.split(rem_idx, rem_y, rem_g))

    target_val_n = max(1, int(round(n * val_ratio)))
    best_inner = min(
        folds_inner,
        key=lambda tr_va: (
            0 if len(np.unique(rem_y[tr_va[1]])) == len(np.unique(rem_y)) else 1,
            abs(len(tr_va[1]) - target_val_n),
        ),
    )
    train_sub_pos, val_sub_pos = best_inner

    train_idx = [int(idx_arr[train_val_pos[i]]) for i in train_sub_pos]
    val_idx = [int(idx_arr[train_val_pos[i]]) for i in val_sub_pos]
    test_idx = [int(idx_arr[i]) for i in test_pos]

    # Strict post-condition verification: zero source-group overlap across splits
    g_train = {groups[i] for i in train_idx}
    g_val = {groups[i] for i in val_idx}
    g_test = {groups[i] for i in test_idx}
    assert not (g_train & g_val), f"Group leakage detected between train & val: {g_train & g_val}"
    assert not (g_train & g_test), f"Group leakage detected between train & test: {g_train & g_test}"
    assert not (g_val & g_test), f"Group leakage detected between val & test: {g_val & g_test}"

    return train_idx, val_idx, test_idx


def build_group_split_manifest(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Constructs a non-destructive StratifiedGroupKFold split manifest from scanned & grouped records.
    """
    if not records:
        return {"status": "empty", "splits": {"train": [], "val": [], "test": []}}

    indices = list(range(len(records)))
    labels = [r["label"] for r in records]
    groups = [r["source_group"] for r in records]

    try:
        train_i, val_i, test_i = stratified_group_train_val_test_split(
            indices, labels, groups, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    except ValueError as exc:
        return {
            "status": "insufficient_source_groups",
            "error": str(exc),
            "total_groups": len(set(groups)),
            "splits": {"train": [], "val": [], "test": []},
        }

    def summarize(idxs: List[int]) -> Dict[str, Any]:
        c = Counter(records[i]["class_name"] for i in idxs)
        gs = sorted({records[i]["source_group"] for i in idxs})
        return {
            "count": len(idxs),
            "real": int(c.get("real", 0)),
            "fake": int(c.get("fake", 0)),
            "groups": gs,
            "items": [
                {
                    "path": records[i]["path"],
                    "rel_path": records[i]["rel_path"],
                    "label": int(records[i]["label"]),
                    "class_name": records[i]["class_name"],
                    "source_group": records[i]["source_group"],
                }
                for i in idxs
            ],
        }

    return {
        "status": "valid_group_disjoint_split",
        "strategy": "StratifiedGroupKFold",
        "seed": seed,
        "group_overlap_count": 0,
        "summary": {
            "train": {k: v for k, v in summarize(train_i).items() if k != "items"},
            "val": {k: v for k, v in summarize(val_i).items() if k != "items"},
            "test": {k: v for k, v in summarize(test_i).items() if k != "items"},
        },
        "splits": {
            "train": summarize(train_i)["items"],
            "val": summarize(val_i)["items"],
            "test": summarize(test_i)["items"],
        },
    }


def run_dataset_audit(
    data_dir: str = "data",
    output_path: str = "results/dataset_audit.json",
    manifest_output_path: Optional[str] = "results/group_split_manifest.json",
    expected_manifest_paths: Optional[Sequence[str]] = None,
    phash_threshold: int = 4,
    anchor_max_hamming: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs complete non-destructive dataset integrity, duplicate, pHash near-duplicate,
    and source-group leakage audit, writing structured results to output_path.
    Never modifies or deletes source image files.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

    scan = scan_dataset_images(data_path, manifest_paths=expected_manifest_paths)
    records = scan["records"]

    exact_dup_report = detect_exact_duplicates(records)
    near_dup_report = detect_near_duplicates(records, phash_threshold=phash_threshold)
    source_group_report = assign_source_groups(records, data_path, anchor_max_hamming=anchor_max_hamming)

    has_leakage = (
        exact_dup_report["cross_split_duplicate_group_count"] > 0
        or near_dup_report["cross_split_near_duplicate_image_count"] > 0
        or source_group_report["cross_split_leaked_group_count"] > 0
    )

    audit_payload: Dict[str, Any] = {
        "data_dir": str(data_path.resolve()),
        "non_destructive_guarantee": "Audit scanned all files read-only; no files were moved, altered, or deleted.",
        "total_image_count": len(records) + len(scan["corrupted_files"]),
        "valid_image_count": len(records),
        "real_fake_class_distribution": scan["class_counts"],
        "train_val_test_distribution": scan["split_counts"],
        "split_class_distribution": scan["split_class_distribution"],
        "duplicate_image_count": exact_dup_report["duplicate_extra_copy_count"],
        "duplicate_total_involved_files": exact_dup_report["total_files_in_duplicate_groups"],
        "near_duplicate_image_count": near_dup_report["near_duplicate_image_count"],
        "corrupted_image_count": len(scan["corrupted_files"]),
        "missing_file_count": len(scan["missing_files"]),
        "resolution_distribution": scan["resolution_distribution"],
        "color_mode_distribution": scan["mode_distribution"],
        "exact_duplicate_analysis": exact_dup_report,
        "perceptual_hash_analysis": near_dup_report,
        "source_group_analysis": source_group_report,
        "corrupted_files": scan["corrupted_files"],
        "missing_files": scan["missing_files"],
        "leakage_detected": bool(has_leakage),
        "requires_explicit_confirmation_for_cleanup": True,
        "cleanup_recommendations": [
            f"Review and remove or relabel {exact_dup_report['cross_label_conflict_group_count']} exact cross-label SHA-256 duplicate group(s) (requires confirmation)."
            if exact_dup_report["cross_label_conflict_group_count"] > 0
            else "No cross-label exact duplicates detected.",
            f"Deduplicate {exact_dup_report['cross_split_duplicate_group_count']} exact cross-split duplicate group(s) ({exact_dup_report['duplicate_extra_copy_count']} extra copies) across train/val/test."
            if exact_dup_report["cross_split_duplicate_group_count"] > 0
            else "No exact cross-split duplicates detected.",
            f"Re-split dataset using StratifiedGroupKFold ({source_group_report['cross_split_leaked_group_count']} source group(s) currently span multiple splits) and prefer multi-identity datasets (FaceForensics++, Celeb-DF, DFDC)."
            if source_group_report["cross_split_leaked_group_count"] > 0
            else "No cross-split source-group leakage detected.",
        ],
    }

    if manifest_output_path:
        manifest = build_group_split_manifest(records, val_ratio=0.2, test_ratio=0.2, seed=seed)
        m_out = Path(manifest_output_path)
        m_out.parent.mkdir(parents=True, exist_ok=True)
        with open(m_out, "w") as f:
            json.dump(manifest, f, indent=2)
        audit_payload["group_split_manifest_path"] = str(m_out)
        audit_payload["group_split_manifest_summary"] = manifest.get("summary", {})
        audit_payload["group_split_status"] = manifest.get("status", "unknown")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(audit_payload, f, indent=2)

    return audit_payload


def main():
    parser = argparse.ArgumentParser(
        description="Non-destructive SHA-256 + pHash Dataset Integrity & Source-Leakage Audit"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/dataset_audit.json")
    parser.add_argument("--manifest", type=str, default="results/group_split_manifest.json")
    parser.add_argument("--phash_threshold", type=int, default=4)
    parser.add_argument("--anchor_max_hamming", type=int, default=12)
    args = parser.parse_args()

    report = run_dataset_audit(
        data_dir=args.data_dir,
        output_path=args.output,
        manifest_output_path=args.manifest,
        phash_threshold=args.phash_threshold,
        anchor_max_hamming=args.anchor_max_hamming,
    )
    print(f"[✓] Dataset Audit Complete -> {args.output}")
    print(f"    Total images scanned   : {report['total_image_count']} (Valid: {report['valid_image_count']}, Corrupt: {report['corrupted_image_count']}, Missing: {report['missing_file_count']})")
    print(f"    Exact dup extra copies : {report['duplicate_image_count']} (Cross-split groups: {report['exact_duplicate_analysis']['cross_split_duplicate_group_count']}, Label conflicts: {report['exact_duplicate_analysis']['cross_label_conflict_group_count']})")
    print(f"    pHash near-dups (<= {args.phash_threshold}): {report['near_duplicate_image_count']} images ({report['perceptual_hash_analysis']['cross_split_near_duplicate_image_count']} cross-split)")
    print(f"    Source groups detected : {report['source_group_analysis']['total_source_groups']} (Cross-split leaked groups: {report['source_group_analysis']['cross_split_leaked_group_count']})")
    print(f"    Group split manifest   : {report.get('group_split_manifest_path')} ({report.get('group_split_status')})")


if __name__ == "__main__":
    main()
