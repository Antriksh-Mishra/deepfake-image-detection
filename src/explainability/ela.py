"""
Error Level Analysis (ELA) Module.
Detects image manipulations by comparing the error between an original image
and its re-saved JPEG version at a specified quality level.
Manipulated or synthetic regions typically display different compression error levels.
"""

import io
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


ELA_FORENSIC_LIMITATIONS: List[str] = [
    "Repeat / Social-Media Recompression Washout: Re-saving an image multiple times (e.g., messaging apps or web uploads) homogenizes JPEG quantization residuals and can erase prior splicing traces.",
    "Natural Edge & High-Frequency Texture Responses: Sharp edges (hair strands, eyelashes, glasses, striped textiles) and lossless PNG-to-JPEG conversions naturally yield elevated ELA brightness even in authentic photos.",
    "Uniformity of Single-Pass AI Generations: Whole-frame GAN or Diffusion images often share a single compression history across the entire canvas, producing low spatial ELA discrepancy despite being synthetic.",
    "Non-Proof Advisory Scope: ELA reflects localized compression error variance rather than semantic truth and must never be presented as standalone proof of forgery.",
]


class ErrorLevelAnalysis:
    def __init__(self, quality: int = 90, scale: int = 15):
        """
        Args:
            quality: JPEG re-compression quality level (default: 90).
            scale: Contrast enhancement factor for error visualization.
        """
        self.quality = quality
        self.scale = scale

    def compute_ela(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """
        Computes ELA on a PIL Image.
        
        Returns:
            ela_image: PIL Image with enhanced error levels.
            mean_error: Average per-pixel error magnitude.
        """
        # Ensure RGB
        img_rgb = image.convert("RGB")
        
        # Save to in-memory JPEG buffer
        buffer = io.BytesIO()
        img_rgb.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        
        # Load re-compressed image
        resaved_img = Image.open(buffer).convert("RGB")
        
        # Calculate pixel-wise absolute difference
        diff = ImageChops.difference(img_rgb, resaved_img)
        
        # Calculate statistical mean error
        diff_np = np.asarray(diff, dtype=np.float32)
        mean_error = float(np.mean(diff_np))
        
        # Scale difference for human visibility
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            scale_factor = 1.0
        else:
            scale_factor = min(self.scale, 255.0 / max_diff)
        
        enhancer = ImageEnhance.Brightness(diff)
        ela_image = enhancer.enhance(scale_factor)
        
        return ela_image, mean_error

    def _render_color_heatmap(self, diff_np_gray: np.ndarray, scale_factor: float) -> Image.Image:
        """Converts raw grayscale pixel-difference array to an Inferno RGB ELA heatmap."""
        scaled = np.clip(diff_np_gray * scale_factor, 0.0, 255.0).astype(np.uint8)
        if HAS_CV2:
            colored_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(colored_rgb)
        return Image.fromarray(np.stack([scaled] * 3, axis=-1))

    @staticmethod
    def get_limitations() -> List[str]:
        """Returns explicit forensic science caveats for Error Level Analysis."""
        return list(ELA_FORENSIC_LIMITATIONS)

    def _compute_single_quality_ela(self, img_rgb: Image.Image, quality: int) -> Dict[str, Any]:
        buffer = io.BytesIO()
        img_rgb.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        resaved_img = Image.open(buffer).convert("RGB")

        diff = ImageChops.difference(img_rgb, resaved_img)
        diff_np = np.asarray(diff, dtype=np.float32)
        mean_error = float(np.mean(diff_np))
        max_error = float(np.max(diff_np))
        std_error = float(np.std(diff_np))

        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale_factor = min(self.scale, 255.0 / max_diff) if max_diff > 0 else 1.0
        ela_image = ImageEnhance.Brightness(diff).enhance(scale_factor)

        diff_gray = np.mean(diff_np, axis=2)
        ela_heatmap_color = self._render_color_heatmap(diff_gray, scale_factor)

        h, w, _ = diff_np.shape
        mid_h, mid_w = max(1, h // 2), max(1, w // 2)
        q_tl = float(np.mean(diff_np[:mid_h, :mid_w]))
        q_tr = float(np.mean(diff_np[:mid_h, mid_w:]))
        q_bl = float(np.mean(diff_np[mid_h:, :mid_w]))
        q_br = float(np.mean(diff_np[mid_h:, mid_w:]))
        quad_stds = [q_tl, q_tr, q_bl, q_br]
        quad_variance = float(np.var(quad_stds))
        quad_discrepancy_ratio = float((max(quad_stds) - min(quad_stds)) / (np.mean(quad_stds) + 1e-6))

        # Local 16x16 block error inconsistency (block coefficient of variation)
        bh, bw = max(1, h // 16), max(1, w // 16)
        block_means = []
        for r in range(0, h - bh + 1, bh):
            for c in range(0, w - bw + 1, bw):
                block_means.append(float(np.mean(diff_gray[r:r + bh, c:c + bw])))
        block_cv = float(np.std(block_means) / (np.mean(block_means) + 1e-6)) if block_means else 0.0

        return {
            "quality": int(quality),
            "resaved_image": resaved_img,
            "ela_image": ela_image,
            "ela_heatmap_color": ela_heatmap_color,
            "mean_error": mean_error,
            "max_error": max_error,
            "std_error": std_error,
            "quadrant_errors": {
                "top_left": q_tl,
                "top_right": q_tr,
                "bottom_left": q_bl,
                "bottom_right": q_br,
            },
            "quad_variance": quad_variance,
            "quad_discrepancy_ratio": quad_discrepancy_ratio,
            "block_inconsistency_cv": round(block_cv, 4),
            "scale_factor": scale_factor,
            "diff_np_gray": diff_gray,
        }

    def compute_ela_detailed(
        self,
        image: Image.Image,
        container_format: Optional[str] = None,
        object_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Computes comprehensive ELA with pixel intensity histograms,
        quadrant/block compression inconsistency metrics, container format awareness,
        and explicit forensic limitations.
        """
        fmt = (container_format or getattr(image, "format", None) or "JPEG").upper()
        is_lossless = fmt in ("PNG", "TIFF", "BMP", "WEBP_LOSSLESS")

        img_rgb = image.convert("RGB")
        single = self._compute_single_quality_ela(img_rgb, quality=self.quality)
        diff_gray = single["diff_np_gray"]
        gray_flat = diff_gray.ravel()
        hist_counts, bin_edges = np.histogram(gray_flat, bins=15, range=(0, max(15.0, single["max_error"])))

        q_ratio = single["quad_discrepancy_ratio"]
        b_cv = single["block_inconsistency_cv"]

        # Foreground-aware error calculation if object bounding box is provided
        foreground_mean = None
        background_mean = None
        h, w = diff_gray.shape
        if object_bbox is not None:
            ox, oy, ow, oh = object_bbox
            ox2, oy2 = min(w, ox + ow), min(h, oy + oh)
            if ow > 0 and oh > 0 and (ow * oh) < (0.90 * w * h):
                fg_mask = np.zeros((h, w), dtype=bool)
                fg_mask[oy:oy2, ox:ox2] = True
                foreground_mean = float(np.mean(diff_gray[fg_mask]))
                background_mean = float(np.mean(diff_gray[~fg_mask])) if np.any(~fg_mask) else 0.0

        if is_lossless:
            inconsistency_level = "Lossless Source Quantization"
            inconsistency_summary = (
                f"Source container format is lossless {fmt}. ELA residuals (mean={single['mean_error']:.2f}) reflect initial JPEG "
                "quantization distortion across sharp boundaries/textures, rather than multi-generation JPEG recompression."
            )
        elif q_ratio > 0.55 or b_cv > 0.65:
            inconsistency_level = "Elevated Localized Inconsistency"
            inconsistency_summary = (
                f"Notable spatial compression gradient across quadrants (ratio={q_ratio:.3f}) "
                f"and 16x16 blocks (CV={b_cv:.3f}). May indicate localized editing OR high-contrast edges/textures."
            )
        elif q_ratio > 0.25 or b_cv > 0.35:
            inconsistency_level = "Moderate Regional Variance"
            inconsistency_summary = (
                f"Moderate quantization variation across image regions (quad ratio={q_ratio:.3f}, block CV={b_cv:.3f}); "
                "typical of mixed scene textures or light recompression."
            )
        else:
            inconsistency_level = "Low / Uniform Compression"
            inconsistency_summary = (
                f"Relatively uniform JPEG error residuals (quad ratio={q_ratio:.3f}, block CV={b_cv:.3f}). "
                "Consistent with single-pass capture OR homogeneously re-encoded media."
            )

        return {
            "resaved_image": single["resaved_image"],
            "ela_image": single["ela_image"],
            "ela_heatmap_color": single["ela_heatmap_color"],
            "mean_error": single["mean_error"],
            "max_error": single["max_error"],
            "std_error": single["std_error"],
            "hist_counts": hist_counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "quadrant_errors": single["quadrant_errors"],
            "quad_variance": single["quad_variance"],
            "quad_discrepancy_ratio": q_ratio,
            "block_inconsistency_cv": b_cv,
            "inconsistency_level": inconsistency_level,
            "inconsistency_summary": inconsistency_summary,
            "recompression_quality": self.quality,
            "scale_factor": single["scale_factor"],
            "diff_np_gray": diff_gray,
            "container_format": fmt,
            "is_lossless_source": is_lossless,
            "foreground_mean_error": round(foreground_mean, 4) if foreground_mean is not None else None,
            "background_mean_error": round(background_mean, 4) if background_mean is not None else None,
            "limitations": self.get_limitations(),
        }

    def compute_recompression_comparison(
        self,
        image: Image.Image,
        qualities: Sequence[int] = (95, 90, 75),
    ) -> List[Dict[str, Any]]:
        """
        Generates side-by-side JPEG recompression comparisons across multiple quality factors
        to help examiners distinguish true quantization anomalies from baseline quality sensitivity.
        """
        img_rgb = image.convert("RGB")
        comparisons: List[Dict[str, Any]] = []
        for q in qualities:
            res = self._compute_single_quality_ela(img_rgb, quality=int(q))
            comparisons.append(
                {
                    "quality": res["quality"],
                    "resaved_image": res["resaved_image"],
                    "ela_image": res["ela_image"],
                    "ela_heatmap_color": res["ela_heatmap_color"],
                    "mean_error": round(res["mean_error"], 3),
                    "max_error": round(res["max_error"], 3),
                    "quad_discrepancy_ratio": round(res["quad_discrepancy_ratio"], 3),
                    "block_inconsistency_cv": res["block_inconsistency_cv"],
                }
            )
        return comparisons
