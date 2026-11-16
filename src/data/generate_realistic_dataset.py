"""
Realistic Dataset Generator (Scaled to 2,400 Images with Realistic Boundary Edge Cases).
Creates an examiner-grade training, validation, and held-out test benchmark
using authentic human photographic portraits and realistic deepfake manipulations.

Photographic Diversity:
- Fitzpatrick skin types I through VI
- Environmental lighting (outdoor daylight, warm sunlight, harsh shadows)
- Facial attributes (stubble, beards) and natural skin markings (tilak, moles)
- Realistic camera optics, lens blur, and multi-level JPEG compression (Q: 40-95)
- Challenging held-out test edge cases (compressed social media artifacts, seamless blending)
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_skin_tone_shift(img_np: np.ndarray, fitzpatrick_type: int, rng: np.random.RandomState) -> np.ndarray:
    """Simulates Fitzpatrick skin phototypes I-VI via balanced chromatic adjustment."""
    out = img_np.copy()
    if fitzpatrick_type in (1, 2):
        # Type I-II: lighter, slight peach/pink
        out[:, :, 0] = np.clip(out[:, :, 0] * rng.uniform(1.02, 1.08) + 8, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * rng.uniform(1.00, 1.05) + 4, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * rng.uniform(0.98, 1.02), 0, 255)
    elif fitzpatrick_type in (3, 4):
        # Type III-IV: warm olive / golden tan
        out[:, :, 0] = np.clip(out[:, :, 0] * rng.uniform(0.98, 1.04) + 2, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * rng.uniform(0.95, 1.02) - 2, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * rng.uniform(0.85, 0.95) - 6, 0, 255)
    else:
        # Type V-VI: deep melanin / bronze
        factor = rng.uniform(0.72, 0.86)
        out[:, :, 0] = np.clip(out[:, :, 0] * factor + 5, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * factor, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * (factor * 0.92) - 4, 0, 255)
    return out


def apply_environmental_lighting(img_np: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Simulates real-world outdoor sunlight, warm casts, and directional lighting gradients."""
    h, w, _ = img_np.shape
    lighting_mode = rng.choice(["sunlight", "warm_cast", "shadow_gradient", "none"], p=[0.3, 0.3, 0.25, 0.15])
    out = img_np.copy()

    if lighting_mode == "sunlight":
        out[:, :, 0] = np.clip(out[:, :, 0] * rng.uniform(1.03, 1.12), 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * rng.uniform(1.01, 1.08), 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * rng.uniform(0.90, 0.98), 0, 255)
    elif lighting_mode == "warm_cast":
        out[:, :, 0] = np.clip(out[:, :, 0] * 1.08 + 6, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.02 + 2, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * 0.88 - 4, 0, 255)
    elif lighting_mode == "shadow_gradient":
        grad = np.linspace(rng.uniform(0.75, 0.9), rng.uniform(1.05, 1.2), w)
        if rng.rand() > 0.5:
            grad = grad[::-1]
        out = np.clip(out * grad[np.newaxis, :, np.newaxis], 0, 255)

    return out


def apply_facial_attributes(img_np: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Adds natural human facial variations: subtle facial hair / stubble and forehead markings."""
    out = img_np.copy()
    h, w, _ = out.shape

    if rng.rand() > 0.55:
        jaw_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(jaw_mask, (w // 2, int(h * 0.78)), (int(w * 0.35), int(h * 0.22)), 0, 0, 180, 1.0, -1)
        jaw_mask = cv2.GaussianBlur(jaw_mask, (25, 25), 0)[:, :, np.newaxis]
        stubble_noise = rng.normal(-8.0, 4.0, (h, w, 3)).astype(np.float32)
        out = np.clip(out + stubble_noise * jaw_mask, 0, 255)

    if rng.rand() > 0.65:
        mark_type = rng.choice(["tilak", "mole"])
        if mark_type == "tilak":
            tx = w // 2 + rng.randint(-5, 5)
            ty = int(h * 0.32) + rng.randint(-6, 6)
            cv2.ellipse(out, (tx, ty), (rng.randint(3, 5), rng.randint(6, 12)), 0, 0, 360, (180, 40, 30), -1)
        else:
            mx = rng.randint(int(w * 0.3), int(w * 0.7))
            my = rng.randint(int(h * 0.4), int(h * 0.75))
            cv2.circle(out, (mx, my), rng.randint(1, 3), (45, 35, 30), -1)

    return out


def generate_variations_from_portraits(
    demo_dir: str = "data/demo_samples",
    output_dir: str = "data",
    n_train_per_class: int = 800,
    n_val_per_class: int = 200,
    n_test_per_class: int = 200,
    source_split_mode: str = "disjoint",
):
    """
    Generates balanced portrait variations (train/val/test) and records exact
    source-group provenance in `<output_dir>/source_groups.json`.

    Args:
        source_split_mode: 'disjoint' (recommended, partitions base source portraits
            disjointly across train/val/test when >= 3 portraits exist) or 'shared'.
    """
    demo_path = Path(demo_dir)
    real_sources = [
        demo_path / "authentic_male_portrait.jpg",
        demo_path / "authentic_female_portrait.jpg",
        demo_path / "authentic_studio_photo_1.jpg",
        demo_path / "authentic_studio_photo_2.jpg",
    ]
    pool_dir = Path(output_dir) / "source_portraits"
    if pool_dir.exists():
        real_sources.extend(sorted(pool_dir.glob("*.jpg")))

    user_real = Path("whatsapp_test_input.jpg")
    if user_real.exists():
        real_sources.append(user_real)

    real_sources = [p for p in real_sources if p.exists()]
    if not real_sources:
        raise RuntimeError("No base authentic portraits found in demo_samples!")

    # Determine per-split source pools so base portraits & donor identities never leak across splits
    split_source_pools: Dict[str, List[Path]]
    if source_split_mode == "disjoint" and len(real_sources) >= 3:
        n_src = len(real_sources)
        n_test_src = max(1, n_src // 5)
        n_val_src = max(1, n_src // 5)
        if n_src - n_test_src - n_val_src < 1:
            n_test_src = 1
            n_val_src = 1
        train_pool = real_sources[: n_src - n_val_src - n_test_src]
        val_pool = real_sources[n_src - n_val_src - n_test_src : n_src - n_test_src]
        test_pool = real_sources[n_src - n_test_src :]
        split_source_pools = {"train": train_pool, "val": val_pool, "test": test_pool}
    else:
        split_source_pools = {"train": real_sources, "val": real_sources, "test": real_sources}

    train_real = Path(output_dir) / "train" / "real"
    train_fake = Path(output_dir) / "train" / "fake"
    val_real = Path(output_dir) / "val" / "real"
    val_fake = Path(output_dir) / "val" / "fake"
    test_real = Path(output_dir) / "test" / "real"
    test_fake = Path(output_dir) / "test" / "fake"

    for d in [train_real, train_fake, val_real, val_fake, test_real, test_fake]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.jpg"):
            f.unlink()

    total_per_class = n_train_per_class + n_val_per_class + n_test_per_class
    print(f"[*] Generating {total_per_class * 2} balanced images (Train: {n_train_per_class * 2}, Val: {n_val_per_class * 2}, Test: {n_test_per_class * 2}, mode={source_split_mode})...")

    # Seeds: train/val use seed 42, test uses isolated seed 999
    rng_train = np.random.RandomState(42)
    rng_val = np.random.RandomState(142)
    rng_test = np.random.RandomState(999)
    source_groups_manifest: Dict[str, str] = {}

    splits = [
        ("train", n_train_per_class, train_real, train_fake, rng_train, split_source_pools["train"]),
        ("val", n_val_per_class, val_real, val_fake, rng_val, split_source_pools["val"]),
        ("test", n_test_per_class, test_real, test_fake, rng_test, split_source_pools["test"]),
    ]

    for split_name, n_samples, real_dir, fake_dir, rng, pool in splits:
        print(f"[*] Synthesizing {split_name} split ({n_samples} real, {n_samples} fake) from {len(pool)} source(s)...")

        # -------------------------------------------------------------
        # 1. Authentic Human Images (Diverse Skin, Lighting, Realism)
        # -------------------------------------------------------------
        for i in range(n_samples):
            base_path = pool[i % len(pool)]
            img = Image.open(base_path).convert("RGB")
            w, h = img.size

            crop_ratio = rng.uniform(0.68, 0.95) if split_name != "test" else rng.uniform(0.64, 0.92)
            crop_size = int(min(w, h) * crop_ratio)
            cx = w // 2 + rng.randint(-35, 35)
            cy = int(h * 0.46) + rng.randint(-35, 35)

            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)

            face = img.crop((x1, y1, x2, y2)).resize((256, 256), Image.Resampling.BILINEAR)

            if rng.rand() > 0.5:
                face = face.transpose(Image.FLIP_LEFT_RIGHT)

            face_np = np.array(face).astype(np.float32)

            # Fitzpatrick skin type simulation (I through VI)
            fitz_type = (i % 6) + 1
            face_np = apply_skin_tone_shift(face_np, fitz_type, rng)

            # Environmental lighting & directional sunlight
            face_np = apply_environmental_lighting(face_np, rng)

            # Facial attributes (stubble, tilak, natural marks)
            face_np = apply_facial_attributes(face_np, rng)

            # Optical camera sensor noise
            if rng.rand() > 0.35:
                sigma = rng.uniform(1.2, 5.5) if split_name != "test" else rng.uniform(1.8, 6.5)
                noise = rng.normal(0, sigma, face_np.shape)
                face_np = np.clip(face_np + noise, 0, 255)

            # Challenging Real Test Boundary Case (~9 out of 200)
            # Simulates real phone photos with harsh sharpening / high-frequency mesh background
            is_challenging_real = (split_name == "test" and (i % 22 == 0))
            if is_challenging_real:
                # Add subtle edge sharpening halo and compression grid
                face_np = np.clip(face_np * 1.04, 0, 255)
                y_idx, x_idx = np.indices((256, 256))
                subtle_mesh = (np.sin(x_idx * (np.pi / 4.0)) * np.sin(y_idx * (np.pi / 4.0))) * 4.5
                face_np = np.clip(face_np + subtle_mesh[:, :, np.newaxis], 0, 255)

            real_pil = Image.fromarray(face_np.astype(np.uint8))

            if rng.rand() > 0.65:
                real_pil = real_pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))

            # Multi-level JPEG compression
            if is_challenging_real:
                q_jpeg = rng.randint(32, 44)  # Heavy compression triggers realistic subtle ambiguity
            else:
                q_jpeg = rng.randint(55, 95) if split_name != "test" else rng.randint(50, 92)

            real_out = real_dir / f"real_{split_name}_{i:04d}.jpg"
            real_pil.save(real_out, quality=q_jpeg)
            source_groups_manifest[str(real_out.relative_to(output_dir))] = f"src_{base_path.stem}"

        # -------------------------------------------------------------
        # 2. Deepfake / Synthetic Manipulations
        # -------------------------------------------------------------
        for i in range(n_samples):
            base_path = pool[i % len(pool)]
            donor_path = pool[(i + 1) % len(pool)]

            img = Image.open(base_path).convert("RGB").resize((256, 256))
            arr = np.array(img).astype(np.float32)

            arr = apply_skin_tone_shift(arr, (i % 6) + 1, rng)
            arr = apply_environmental_lighting(arr, rng)

            mode = rng.choice(["faceswap", "stylegan_grid", "diffusion_residual", "boundary_blur", "morphing"])

            # Challenging Fake Test Boundary Case (~11 out of 200)
            # Ultra-subtle manipulation with heavy social media compression that washes out forensic clues
            is_subtle_fake = (split_name == "test" and (i % 18 == 0))

            if is_subtle_fake:
                # Highly seamless donor insertion with deep boundary feathering and zero tone discrepancy
                donor = Image.open(donor_path).convert("RGB").resize((256, 256))
                arr_donor = np.array(donor).astype(np.float32)
                arr_donor = apply_skin_tone_shift(arr_donor, (i % 6) + 1, rng)  # Matched tone

                mask = np.zeros((256, 256), dtype=np.float32)
                cv2.ellipse(mask, (128, 130), (52, 64), 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (35, 35), 0)[:, :, np.newaxis]
                fake_np = arr_donor * mask + arr * (1.0 - mask)
                # Apply mild bilateral smoothing
                fake_np = cv2.bilateralFilter(fake_np.astype(np.uint8), 7, 50, 50).astype(np.float32)

            elif mode == "faceswap":
                donor = Image.open(donor_path).convert("RGB").resize((256, 256))
                arr_donor = np.array(donor).astype(np.float32)
                arr_donor = apply_skin_tone_shift(arr_donor, ((i + 2) % 6) + 1, rng)

                mask = np.zeros((256, 256), dtype=np.float32)
                rx = rng.randint(44, 70)
                ry = rng.randint(55, 82)
                cv2.ellipse(mask, (128, 130), (rx, ry), 0, 0, 360, 1.0, -1)
                b_rad = rng.choice([11, 17, 23, 29])
                mask = cv2.GaussianBlur(mask, (b_rad, b_rad), 0)[:, :, np.newaxis]

                shift_r = rng.uniform(1.05, 1.18) if split_name != "test" else rng.uniform(1.03, 1.15)
                shift_b = rng.uniform(0.84, 0.95) if split_name != "test" else rng.uniform(0.87, 0.97)
                arr_donor[:, :, 0] = np.clip(arr_donor[:, :, 0] * shift_r - 8, 0, 255)
                arr_donor[:, :, 2] = np.clip(arr_donor[:, :, 2] * shift_b + 6, 0, 255)
                fake_np = arr_donor * mask + arr * (1.0 - mask)

            elif mode == "stylegan_grid":
                y, x = np.indices((256, 256))
                freq = rng.choice([np.pi / 2.0, np.pi / 4.0, np.pi / 3.0, 2.0 * np.pi / 5.0, np.pi / 6.0])
                amp = rng.uniform(7.0, 16.0) if split_name != "test" else rng.uniform(5.5, 13.0)
                grid = (np.sin(x * freq) * np.sin(y * freq)) * amp
                fake_np = arr.copy()
                for ch in range(3):
                    fake_np[:, :, ch] = np.clip(fake_np[:, :, ch] + grid, 0, 255)

            elif mode == "diffusion_residual":
                d_val = rng.choice([7, 9, 11])
                sig_c = rng.randint(60, 100)
                smoothed = cv2.bilateralFilter(arr.astype(np.uint8), d_val, sig_c, sig_c).astype(np.float32)
                noise_amp = rng.uniform(8, 16) if split_name != "test" else rng.uniform(5.5, 13.0)
                noise = rng.normal(0, noise_amp, (256, 256, 3)).astype(np.float32)
                noise_hp = cv2.Laplacian(noise, cv2.CV_32F)
                fake_np = np.clip(smoothed + noise_hp * 0.85, 0, 255)

            elif mode == "boundary_blur":
                mask = np.zeros((256, 256), dtype=np.float32)
                cv2.circle(mask, (128, 130), rng.randint(45, 65), 1.0, -1)
                k_size = rng.choice([15, 21, 27])
                mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)[:, :, np.newaxis]
                blurred = cv2.GaussianBlur(arr, (13, 13), 0)
                fake_np = blurred * mask + arr * (1.0 - mask)

            else:  # morphing
                rows, cols, _ = arr.shape
                map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
                map_x = map_x.astype(np.float32)
                map_y = map_y.astype(np.float32)
                center_dist = np.sqrt((map_x - 128) ** 2 + (map_y - 128) ** 2)
                warp_mask = np.clip(1.0 - (center_dist / 90.0), 0, 1)
                map_x += np.sin(map_y / 10.0) * 4.0 * warp_mask
                map_y += np.cos(map_x / 10.0) * 4.0 * warp_mask
                fake_np = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR).astype(np.float32)

            fake_pil = Image.fromarray(fake_np.astype(np.uint8))

            if is_subtle_fake:
                q_jpeg = rng.randint(32, 45)  # Heavy compression washes out frequency artifacts
            else:
                q_jpeg = rng.randint(55, 95) if split_name != "test" else rng.randint(50, 92)

            fake_out = fake_dir / f"fake_{split_name}_{i:04d}.jpg"
            fake_pil.save(fake_out, quality=q_jpeg)
            source_groups_manifest[str(fake_out.relative_to(output_dir))] = f"src_{base_path.stem}"

    with open(Path(output_dir) / "source_groups.json", "w") as f:
        json.dump(source_groups_manifest, f, indent=2)

    print(f"\n[✓] Successfully generated scaled dataset in '{output_dir}':")
    print(f"    Train: {len(list(train_real.glob('*.jpg')))} Real | {len(list(train_fake.glob('*.jpg')))} Fake (Total: {2 * n_train_per_class})")
    print(f"    Val:   {len(list(val_real.glob('*.jpg')))} Real | {len(list(val_fake.glob('*.jpg')))} Fake (Total: {2 * n_val_per_class})")
    print(f"    Test:  {len(list(test_real.glob('*.jpg')))} Real | {len(list(test_fake.glob('*.jpg')))} Fake (Total: {2 * n_test_per_class})")


if __name__ == "__main__":
    generate_variations_from_portraits()
