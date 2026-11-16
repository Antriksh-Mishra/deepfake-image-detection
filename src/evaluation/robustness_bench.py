"""
Robustness Stress-Testing and Perturbation Benchmark.
Evaluates model resistance against post-processing attacks commonly encountered
in social media distribution: JPEG compression, Gaussian blurring, and sensor noise.
"""

import io
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from src.core.utils import get_device, safe_batched_forward, SPATIAL_TRANSFORM, SPECTRAL_TRANSFORM
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    if std <= 0:
        return image
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0) * 255.0
    return Image.fromarray(noisy_arr.astype(np.uint8))


class RobustnessBenchmark:
    def __init__(self, model: DualDomainSentinelNet, device: torch.device):
        self.model = model
        self.device = device
        self.spectral_analyzer = SpectralAnalyzer(target_size=256)

    def evaluate_sample(self, image: Image.Image) -> float:
        """Returns deepfake probability for a single image."""
        return self.evaluate_batch([image], batch_size=1)[0]

    def evaluate_batch(self, images: List[Image.Image], batch_size: int = 32) -> List[float]:
        """Returns deepfake probabilities for a list of images using batched forward inference."""
        if not images:
            return []
        self.model.eval()
        probs_all: List[float] = []
        eff_batch = max(1, int(batch_size))

        for start in range(0, len(images), eff_batch):
            chunk = images[start:start + eff_batch]
            sp_list = []
            fr_list = []
            for img in chunk:
                rgb = img.convert("RGB")
                sp_list.append(SPATIAL_TRANSFORM(rgb))
                mag_spec, _ = self.spectral_analyzer.compute_fft_magnitude(rgb)
                fr_list.append(SPECTRAL_TRANSFORM(Image.fromarray(mag_spec)))

            sp_batch = torch.stack(sp_list)
            fr_batch = torch.stack(fr_list)
            with torch.no_grad():
                logits, active_dev, _ = safe_batched_forward(
                    self.model,
                    sp_batch,
                    fr_batch,
                    self.device,
                    enable_cpu_fallback=True,
                )
                self.device = active_dev
                probs_all.extend(torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy().tolist())
        return [float(p) for p in probs_all]

    def run_jpeg_sweep(
        self,
        images: List[Image.Image],
        true_labels: List[int],
        qualities: List[int] = [90, 75, 50, 30, 15],
        batch_size: int = 32,
    ) -> Dict[int, float]:
        """Evaluates accuracy across JPEG compression levels using batched inference."""
        if not images:
            return {q: 0.0 for q in qualities}
        accs = {}
        for q in qualities:
            perturbed_imgs = [apply_jpeg_compression(img, quality=q) for img in images]
            probs = self.evaluate_batch(perturbed_imgs, batch_size=batch_size)
            correct = sum(
                1
                for prob, label in zip(probs, true_labels)
                if (1 if prob >= 0.5 else 0) == label
            )
            accs[q] = correct / len(images)
        return accs

    def plot_jpeg_robustness(self, results: Dict[int, float], save_path: str = "robustness_jpeg.png"):
        qualities = list(results.keys())
        accuracies = [results[q] * 100.0 for q in qualities]

        plt.figure(figsize=(7, 5))
        plt.plot(qualities, accuracies, marker="o", color="#059669", lw=2.5, markersize=8)
        plt.gca().invert_xaxis()  # 90 -> 15 (increasing compression severity)
        plt.xlabel("JPEG Quality Factor (Left: High Quality -> Right: Heavy Compression)", fontsize=11)
        plt.ylabel("Detection Accuracy (%)", fontsize=11)
        plt.title("Model Resilience Under JPEG Compression Perturbation", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.ylim([40, 105])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
