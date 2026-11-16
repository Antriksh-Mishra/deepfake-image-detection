"""
Spatial Rich Model (SRM) Noise Residual & Texture Inconsistency Module.

Computes spatial steganalysis noise residuals using 3 standard SRM high-pass filters:
  - F1: 1st-order directional residual (edge/gradient suppression)
  - F2: 2nd-order Laplacian / KB residual (curvature suppression)
  - F3: 3x3 square edge residual (point/seam suppression)

Analyzes:
  1. Localized noise variance and block-wise noise inconsistency (CV)
  2. Texture naturalness and residual kurtosis
  3. Spatial edge and boundary residual energy
  4. Resampling and synthetic noise patterns
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def get_srm_kernels() -> List[np.ndarray]:
    """
    Returns 3 standard 5x5 SRM high-pass filter kernels normalized as float32.
    Each kernel has zero-sum weights to suppress low-frequency image content.
    """
    f1 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 4.0

    f2 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32) / 12.0

    f3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 4.0

    return [f1, f2, f3]


class SpatialSRMAnalyzer:
    """
    Analyzes spatial domain noise residuals directly on RGB / grayscale input images
    to detect localized splices, synthetic generator artifacts, and texture anomalies.
    """

    def __init__(self):
        self.kernels = get_srm_kernels()

    def filter_image(self, gray_np: np.ndarray) -> List[np.ndarray]:
        """
        Applies the 3 SRM filters using reflection padding to prevent boundary artifacts.

        Args:
            gray_np: 2D float32 image array in [0, 255].

        Returns:
            residuals: List of three 2D float32 residual maps.
        """
        img_f = gray_np.astype(np.float32)
        residuals = []
        for k in self.kernels:
            if HAS_CV2:
                # cv2.filter2D with BORDER_REFLECT_101 prevents artificial zero-padding boundary spikes
                res = cv2.filter2D(img_f, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT_101)
            else:
                pad = 2
                padded = np.pad(img_f, pad, mode="reflect")
                h, w = img_f.shape
                res = np.zeros_like(img_f)
                for i in range(5):
                    for j in range(5):
                        res += k[i, j] * padded[i:i + h, j:j + w]
            residuals.append(res)
        return residuals

    def analyze_residuals(self, image: Image.Image) -> Dict[str, Any]:
        """
        Computes comprehensive spatial SRM forensic metrics:
          - Individual F1, F2, F3 residuals
          - Combined absolute noise residual energy
          - Local block-wise noise inconsistency (Coefficient of Variation)
          - Quadrant noise energy distribution
          - Residual kurtosis & variance (unnatural texture signature)
          - High-contrast colormap visualization

        Args:
            image: PIL Image (RGB or L).

        Returns:
            Forensic metrics dictionary.
        """
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape

        residuals = self.filter_image(gray)
        r1, r2, r3 = residuals

        # Absolute noise energy map
        abs_r1 = np.abs(r1)
        abs_r2 = np.abs(r2)
        abs_r3 = np.abs(r3)
        combined_energy = (abs_r1 + abs_r2 + abs_r3) / 3.0

        mean_energy = float(np.mean(combined_energy))
        max_energy = float(np.max(combined_energy))
        std_energy = float(np.std(combined_energy))

        # Quadrant residual energies
        mid_h, mid_w = max(1, h // 2), max(1, w // 2)
        q_tl = float(np.mean(combined_energy[:mid_h, :mid_w]))
        q_tr = float(np.mean(combined_energy[:mid_h, mid_w:]))
        q_bl = float(np.mean(combined_energy[mid_h:, :mid_w]))
        q_br = float(np.mean(combined_energy[mid_h:, mid_w:]))
        quad_energies = {"top_left": q_tl, "top_right": q_tr, "bottom_left": q_bl, "bottom_right": q_br}
        quad_vals = list(quad_energies.values())
        quad_discrepancy = float((max(quad_vals) - min(quad_vals)) / (np.mean(quad_vals) + 1e-6))

        # Block-wise noise inconsistency (16x16 grid)
        bh, bw = max(4, h // 16), max(4, w // 16)
        block_stds = []
        for r in range(0, h - bh + 1, bh):
            for c in range(0, w - bw + 1, bw):
                patch = combined_energy[r:r + bh, c:c + bw]
                block_stds.append(float(np.std(patch)))
        block_cv = float(np.std(block_stds) / (np.mean(block_stds) + 1e-6)) if block_stds else 0.0

        # Residual kurtosis: natural camera sensor noise is sub-Gaussian / Gaussian;
        # synthetic images often show heavy-tailed kurtosis (>6.0) or over-smoothed noise (<2.0)
        norm_r2 = (r2 - np.mean(r2)) / (np.std(r2) + 1e-6)
        kurtosis = float(np.mean(norm_r2 ** 4))

        # Categorize spatial noise consistency
        if block_cv > 0.60 or quad_discrepancy > 0.55:
            consistency_tier = "Inconsistent Spatial Noise"
            summary = (
                f"Elevated spatial noise variance across blocks (CV={block_cv:.3f}) and quadrants "
                f"(discrepancy={quad_discrepancy:.3f}). Suggests localized blending, splicing, or uneven generation."
            )
        elif kurtosis > 8.0:
            consistency_tier = "Heavy-Tailed Residual Spike"
            summary = (
                f"Unnatural residual kurtosis ({kurtosis:.1f} vs ~3.0 Gaussian baseline). "
                "Indicates sharp synthetic boundary transitions or intense localized sharpening."
            )
        else:
            consistency_tier = "Uniform Natural Residual"
            summary = (
                f"Uniform spatial noise distribution (block CV={block_cv:.3f}, kurtosis={kurtosis:.2f}). "
                "Consistent with homogeneous camera capture or uniform re-encoding."
            )

        # High-contrast visual representation
        scale = 255.0 / (np.percentile(combined_energy, 98) + 1e-6)
        scaled_energy = np.clip(combined_energy * scale, 0.0, 255.0).astype(np.uint8)

        if HAS_CV2:
            colored_bgr = cv2.applyColorMap(scaled_energy, cv2.COLORMAP_VIRIDIS)
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            heatmap_img = Image.fromarray(colored_rgb)
        else:
            heatmap_img = Image.fromarray(np.stack([scaled_energy] * 3, axis=-1))

        return {
            "mean_energy": round(mean_energy, 4),
            "max_energy": round(max_energy, 4),
            "std_energy": round(std_energy, 4),
            "block_inconsistency_cv": round(block_cv, 4),
            "quadrant_energies": {k: round(v, 4) for k, v in quad_energies.items()},
            "quad_discrepancy_ratio": round(quad_discrepancy, 4),
            "residual_kurtosis": round(kurtosis, 2),
            "consistency_tier": consistency_tier,
            "consistency_summary": summary,
            "heatmap_image": heatmap_img,
            "f1_mean": round(float(np.mean(abs_r1)), 4),
            "f2_mean": round(float(np.mean(abs_r2)), 4),
            "f3_mean": round(float(np.mean(abs_r3)), 4),
        }
