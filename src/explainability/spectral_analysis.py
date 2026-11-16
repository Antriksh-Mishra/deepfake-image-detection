"""
Spectral & Frequency Domain Analysis Module.
Computes 2D Fast Fourier Transform (FFT) magnitude spectra and azimuthal (radial)
energy profiles. Exposes spectral discrepancies and checkerboard artifacts caused
by deconvolution/upsampling in generative networks (GANs and Diffusion Models).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def format_frequency_band(band: Tuple[int, int]) -> str:
    """Formats an anomaly band cleanly as r=X if single radius or r=[X..Y] if a range."""
    b0, b1 = int(band[0]), int(band[1])
    if b0 == b1:
        return f"r={b0}"
    return f"r=[{b0}..{b1}]"


class SpectralAnalyzer:
    def __init__(self, target_size: int = 256):
        self.target_size = target_size
        self._reference_cache: Dict[str, Dict[str, Any]] = {}

    def compute_fft_magnitude(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 2D FFT and shifted logarithmic power spectrum.

        Args:
            image: PIL Image.

        Returns:
            magnitude_spectrum: 2D np.ndarray normalized to [0, 255] uint8.
            raw_log_spectrum: 2D np.ndarray float32 for metric calculations.
        """
        gray = np.array(image.convert("L"), dtype=np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        magnitude = np.abs(fshift)
        raw_log_spectrum = np.log(1.0 + magnitude).astype(np.float32)

        norm_spec = raw_log_spectrum - float(np.min(raw_log_spectrum))
        max_val = float(np.max(norm_spec))
        if max_val > 0:
            norm_spec = (norm_spec / max_val) * 255.0
        magnitude_spectrum = norm_spec.astype(np.uint8)
        return magnitude_spectrum, raw_log_spectrum

    def compute_colormap_log_spectrum(self, raw_log_spectrum: np.ndarray) -> Image.Image:
        """Renders a high-contrast pseudo-color log-magnitude 2D FFT visualization."""
        norm = raw_log_spectrum - float(np.min(raw_log_spectrum))
        max_val = float(np.max(norm))
        gray_u8 = ((norm / max_val) * 255.0).astype(np.uint8) if max_val > 0 else np.zeros_like(norm, dtype=np.uint8)
        if HAS_CV2:
            colored_bgr = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(colored_rgb)
        return Image.fromarray(np.stack([gray_u8] * 3, axis=-1))

    def azimuthal_average(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the 1D Azimuthal (radial) average of a 2D power spectrum.
        Natural images follow a smooth radial power decay; synthetic images can exhibit
        mid/high-frequency deviations and periodic spikes.
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(np.int32)

        max_r = min(center_x, center_y)
        tbin = np.bincount(r.ravel(), spectrum.ravel())
        nr = np.bincount(r.ravel())

        radial_profile = tbin[:max_r] / np.maximum(nr[:max_r], 1)
        radial_frequencies = np.arange(max_r)
        return radial_frequencies, radial_profile

    def analyze_spectral_peaks(
        self,
        radial_frequencies: np.ndarray,
        radial_profile: np.ndarray,
        prominence_factor: float = 1.15,
    ) -> Dict[str, Any]:
        """
        Analyzes 1D azimuthal radial profile for anomalous high-frequency energy spikes
        caused by deconvolution checkerboards and computes anomalous radial zones.
        """
        if len(radial_profile) < 10:
            return {
                "detected_peaks": [],
                "has_deconv_spikes": False,
                "high_freq_ratio": 0.0,
                "spectral_smoothness": 0.0,
                "peak_indices": [],
                "ideal_decay": np.copy(radial_profile),
                "anomalous_radii": [],
                "fitted_decay_slope": 1.69,
            }

        # Adaptive optical decay baseline: fit power-law decay slope alpha to low/mid frequencies (r in [2, 35])
        # In log-log space: log_power = Intercept - alpha * log(1 + r)
        fit_end = min(35, len(radial_profile) - 2)
        fit_r = np.maximum(radial_frequencies[2:fit_end], 1.0)
        fit_p = radial_profile[2:fit_end]
        if len(fit_r) >= 5:
            log_r = np.log1p(fit_r)
            poly = np.polyfit(log_r, fit_p, 1)
            fitted_slope = float(-poly[0])
            # Bound slope within physically valid optical bounds [0.8, 3.5]
            fitted_slope = float(np.clip(fitted_slope, 0.80, 3.50))
            intercept = float(poly[1])
        else:
            fitted_slope = 1.69
            base_val = float(radial_profile[1]) if len(radial_profile) > 1 else 1.0
            intercept = base_val + fitted_slope * float(np.log1p(1.0))

        ideal_decay = np.maximum(0.0, intercept - fitted_slope * np.log1p(np.maximum(radial_frequencies, 0.0)))
        residuals = radial_profile - ideal_decay

        peaks = []
        peak_indices = []
        local_noise_std = float(np.std(np.diff(radial_profile[10:]))) if len(radial_profile) > 15 else 0.1
        min_prominence = max(0.12, 2.0 * local_noise_std)
        for i in range(12, len(radial_profile) - 5):
            val = float(radial_profile[i])
            local_base = 0.5 * (float(radial_profile[i - 1]) + float(radial_profile[i + 1]))
            prominence = val - local_base
            # True spike: rises clearly above immediate neighbors AND exceeds expected decay baseline
            if val > radial_profile[i - 1] and val > radial_profile[i + 1]:
                if (residuals[i] > 0.28 and prominence >= min_prominence) or (val > ideal_decay[i] * prominence_factor and prominence >= min_prominence):
                    peaks.append({
                        "radius": int(radial_frequencies[i]),
                        "power": float(val),
                        "residual": float(residuals[i]),
                        "prominence": float(prominence),
                    })
                    peak_indices.append(int(i))

        mid_idx = min(40, len(radial_profile) - 1)
        high_freq_power = float(np.sum(radial_profile[mid_idx:]))
        total_power = float(np.sum(radial_profile) + 1e-6)
        high_freq_ratio = float(high_freq_power / total_power)

        second_diff = np.diff(radial_profile, n=2)
        spectral_smoothness = float(np.var(second_diff))

        tail_res = residuals[15:] if len(residuals) > 16 else residuals
        excess_thresh = max(0.55, float(np.mean(np.maximum(tail_res, 0.0)) + 2.0 * np.std(tail_res)))
        anomalous_radii = [
            int(radial_frequencies[i])
            for i in range(15, len(radial_frequencies) - 2)
            if (residuals[i] > excess_thresh and radial_profile[i] > ideal_decay[i] * 1.12) or i in peak_indices
        ]

        return {
            "detected_peaks": peaks,
            "has_deconv_spikes": len(peaks) > 0,
            "high_freq_ratio": high_freq_ratio,
            "spectral_smoothness": spectral_smoothness,
            "peak_indices": peak_indices,
            "ideal_decay": ideal_decay,
            "anomalous_radii": anomalous_radii,
            "fitted_decay_slope": fitted_slope,
        }

    def compute_frequency_anomaly_map(
        self,
        raw_log_spectrum: np.ndarray,
        radial_frequencies: np.ndarray,
        spectral_details: Dict[str, Any],
    ) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """
        Highlights 2D FFT annular anomaly regions where mid/high-frequency radial energy
        deviates anomalously above expected monotonic optical decay.
        """
        norm = raw_log_spectrum - float(np.min(raw_log_spectrum))
        max_val = float(np.max(norm))
        base_u8 = ((norm / max_val) * 255.0).astype(np.uint8) if max_val > 0 else np.zeros_like(norm, dtype=np.uint8)
        rgb = np.stack([base_u8, base_u8, base_u8], axis=-1).astype(np.float32)

        anomalous_radii = sorted(set(int(r) for r in spectral_details.get("anomalous_radii", [])))
        anomalous_bands: List[Tuple[int, int]] = []
        if anomalous_radii:
            band_start = anomalous_radii[0]
            prev = anomalous_radii[0]
            for r_val in anomalous_radii[1:]:
                if r_val - prev <= 3:
                    prev = r_val
                else:
                    anomalous_bands.append((band_start, prev))
                    band_start = r_val
                    prev = r_val
            anomalous_bands.append((band_start, prev))

        h, w = raw_log_spectrum.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.indices((h, w))
        radius_grid = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if anomalous_bands:
            band_mask = np.zeros((h, w), dtype=bool)
            for b_s, b_e in anomalous_bands:
                band_mask |= (radius_grid >= max(0, b_s - 1)) & (radius_grid <= (b_e + 1))
            rgb[band_mask, 0] = np.clip(rgb[band_mask, 0] * 0.48 + 235.0 * 0.52, 0, 255)
            rgb[band_mask, 1] = np.clip(rgb[band_mask, 1] * 0.48 + 75.0 * 0.52, 0, 255)
            rgb[band_mask, 2] = np.clip(rgb[band_mask, 2] * 0.48 + 55.0 * 0.52, 0, 255)

        # Also highlight localized off-axis 2D Fourier grid spikes exceeding expected radial log decay
        ideal_decay = spectral_details.get("ideal_decay")
        if ideal_decay is not None and len(ideal_decay) > 0:
            r_int = np.clip(radius_grid.astype(np.int32), 0, len(ideal_decay) - 1)
            expected_2d = ideal_decay[r_int]
            excess_2d = raw_log_spectrum - expected_2d
            off_axis = (radius_grid >= 14.0) & (np.abs(xx - cx) > 3) & (np.abs(yy - cy) > 3)
            if np.any(off_axis):
                off_vals = excess_2d[off_axis]
                thresh_2d = float(np.mean(off_vals) + 2.8 * np.std(off_vals))
                local_peaks = off_axis & (excess_2d > max(thresh_2d, 1.5))
                if np.any(local_peaks):
                    if HAS_CV2:
                        local_peaks = cv2.dilate(local_peaks.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    rgb[local_peaks, 0] = 250.0
                    rgb[local_peaks, 1] = 204.0
                    rgb[local_peaks, 2] = 21.0

        return Image.fromarray(rgb.astype(np.uint8)), anomalous_bands

    def _get_reference_profiles(self, length: int, reference_dir: str = "data/demo_samples") -> Dict[str, Any]:
        """Computes & caches empirical radial profiles from local verified authentic vs deepfake samples."""
        cache_key = f"{reference_dir}:{length}"
        if cache_key in self._reference_cache:
            return self._reference_cache[cache_key]

        ref_path = Path(reference_dir)
        real_profiles: List[np.ndarray] = []
        fake_profiles: List[np.ndarray] = []

        if ref_path.is_dir():
            for p in sorted(ref_path.glob("*.jpg")):
                try:
                    img = Image.open(p).convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
                    _, raw_log = self.compute_fft_magnitude(img)
                    _, prof = self.azimuthal_average(raw_log)
                    if len(prof) != length:
                        prof = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(prof)), prof)
                    if p.name.startswith("authentic_"):
                        real_profiles.append(prof)
                    elif p.name.startswith("deepfake_"):
                        fake_profiles.append(prof)
                except Exception:
                    continue

        if real_profiles:
            real_arr = np.vstack(real_profiles)
            real_mean = np.mean(real_arr, axis=0)
            real_std = np.std(real_arr, axis=0)
        else:
            real_mean = None
            real_std = None

        fake_mean = np.mean(np.vstack(fake_profiles), axis=0) if fake_profiles else None

        res = {
            "available": bool(real_profiles),
            "num_authentic_refs": len(real_profiles),
            "num_suspicious_refs": len(fake_profiles),
            "real_mean": real_mean,
            "real_std": real_std,
            "fake_mean": fake_mean,
        }
        self._reference_cache[cache_key] = res
        return res

    def compare_real_vs_suspicious(
        self,
        radial_frequencies: np.ndarray,
        radial_profile: np.ndarray,
        spectral_details: Dict[str, Any],
        reference_dir: str = "data/demo_samples",
    ) -> Dict[str, Any]:
        """
        Compares the query azimuthal radial profile against theoretical optical 1/f decay
        and empirical local authentic vs suspicious reference spectra.
        """
        n_r = len(radial_profile)
        ideal_decay = spectral_details.get("ideal_decay")
        if ideal_decay is None or len(ideal_decay) != n_r:
            base_val = float(radial_profile[1]) if n_r > 1 else 1.0
            intercept = base_val + 1.69 * float(np.log1p(1.0))
            ideal_decay = np.maximum(0.0, intercept - 1.69 * np.log1p(np.maximum(radial_frequencies, 0.0)))

        refs = self._get_reference_profiles(length=n_r, reference_dir=reference_dir)
        real_mean = refs["real_mean"]
        real_std = refs["real_std"]
        fake_mean = refs["fake_mean"]

        baseline = real_mean if real_mean is not None else ideal_decay
        tail_slice = slice(min(15, n_r), n_r)
        sample_tail = radial_profile[tail_slice]
        base_tail = baseline[tail_slice]

        l1_dev = float(np.mean(np.abs(sample_tail - base_tail))) if len(sample_tail) > 0 else 0.0
        excess_pct = (
            float(np.mean(np.maximum(sample_tail - base_tail, 0.0) / (np.abs(base_tail) + 1e-6)) * 100.0)
            if len(sample_tail) > 0
            else 0.0
        )

        has_spikes = bool(spectral_details.get("has_deconv_spikes", False))
        if has_spikes or excess_pct > 12.0:
            summary = (
                f"Elevated high-frequency radial energy (+{excess_pct:.1f}% vs reference baseline, "
                f"{len(spectral_details.get('detected_peaks', []))} prominent periodic peaks). "
                "Note: heavy sharpening, resizing, or JPEG compression can also introduce spectral ripples."
            )
        else:
            summary = (
                f"Radial log-power tracks monotonic optical decay cleanly (mean deviation {l1_dev:.3f}, "
                f"excess {excess_pct:.1f}% vs reference baseline)."
            )

        return {
            "available": refs["available"],
            "num_authentic_refs": refs["num_authentic_refs"],
            "num_suspicious_refs": refs["num_suspicious_refs"],
            "ideal_decay": np.asarray(ideal_decay, dtype=np.float32),
            "reference_real_mean": np.asarray(real_mean, dtype=np.float32) if real_mean is not None else None,
            "reference_real_std": np.asarray(real_std, dtype=np.float32) if real_std is not None else None,
            "reference_fake_mean": np.asarray(fake_mean, dtype=np.float32) if fake_mean is not None else None,
            "mean_l1_deviation": round(l1_dev, 4),
            "high_freq_excess_pct": round(excess_pct, 2),
            "comparison_summary": summary,
        }
