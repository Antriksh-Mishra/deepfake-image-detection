"""
Utility functions for hardware detection, safe device fallback execution,
CPU/GPU/MPS memory telemetry, configuration loading, reproducibility seeding,
and image-tensor transformations.
"""

import os
import random
import resource
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms


def get_device(requested_device: str = "auto") -> torch.device:
    """
    Selects the optimal hardware acceleration device with safe fallback:
    Apple Silicon MPS -> CUDA GPU -> CPU.
    """
    req = str(requested_device or "auto").strip().lower()
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")

    # Auto mode (or unrecognized fallback)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_accelerator_cache(device: Optional[torch.device] = None) -> None:
    """Releases cached accelerator memory safely on MPS or CUDA without raising."""
    dev_type = device.type if device is not None else None
    try:
        if (dev_type == "cuda" or dev_type is None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if (
            (dev_type == "mps" or dev_type is None)
            and torch.backends.mps.is_available()
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()
    except Exception:
        pass


def safe_batched_forward(
    model: torch.nn.Module,
    spatial_batch: torch.Tensor,
    spectral_batch: torch.Tensor,
    device: torch.device,
    enable_cpu_fallback: bool = True,
) -> Tuple[torch.Tensor, torch.device, bool]:
    """
    Executes a batched forward pass on `device`, automatically falling back to CPU
    if an accelerator RuntimeError (e.g., MPS/CUDA OOM or unsupported op) occurs.

    Returns:
        (logits_tensor, active_device, fallback_triggered)
    """
    active_device = device
    try:
        logits = model(spatial_batch.to(active_device), spectral_batch.to(active_device))
        return logits, active_device, False
    except RuntimeError:
        if not enable_cpu_fallback or active_device.type == "cpu":
            raise
        clear_accelerator_cache(active_device)
        cpu_device = torch.device("cpu")
        model.to(cpu_device)
        logits = model(spatial_batch.to(cpu_device), spectral_batch.to(cpu_device))
        return logits, cpu_device, True


def get_runtime_memory_telemetry(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Collects honest, live process RAM (MB) and accelerator (MPS/CUDA) memory allocation telemetry.
    Uses `psutil` if installed, otherwise falls back cleanly to stdlib `resource.getrusage`.
    """
    active_dev = device if device is not None else get_device("auto")
    dev_type = active_dev.type

    cpu_rss_mb = 0.0
    memory_backend = "unavailable"
    try:
        import psutil  # type: ignore

        cpu_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        memory_backend = "psutil"
    except Exception:
        try:
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports ru_maxrss in bytes; Linux/BSD report kilobytes.
            if sys.platform == "darwin":
                cpu_rss_mb = float(max_rss) / (1024.0 * 1024.0)
            else:
                cpu_rss_mb = float(max_rss) / 1024.0
            memory_backend = "resource.getrusage_peak_rss"
        except Exception:
            cpu_rss_mb = 0.0
            memory_backend = "unavailable"

    accel_alloc_mb = 0.0
    accel_reserved_mb = 0.0
    if dev_type == "cuda" and torch.cuda.is_available():
        try:
            accel_alloc_mb = float(torch.cuda.memory_allocated(active_dev)) / (1024.0 * 1024.0)
            accel_reserved_mb = float(torch.cuda.memory_reserved(active_dev)) / (1024.0 * 1024.0)
        except Exception:
            pass
    elif dev_type == "mps" and torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                accel_alloc_mb = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
            if hasattr(torch.mps, "driver_allocated_memory"):
                accel_reserved_mb = float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)
        except Exception:
            pass

    return {
        "device": dev_type,
        "cpu_rss_mb": round(cpu_rss_mb, 2),
        "accelerator_alloc_mb": round(accel_alloc_mb, 2),
        "accelerator_reserved_mb": round(accel_reserved_mb, 2),
        "memory_backend": memory_backend,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def compute_throughput_fps(item_count: int, elapsed_ms: float) -> float:
    """Computes items processed per second from item count and elapsed milliseconds."""
    if item_count <= 0:
        return 0.0
    elapsed_sec = max(float(elapsed_ms) / 1000.0, 1e-6)
    return round(float(item_count) / elapsed_sec, 2)


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Standard ImageNet normalization transforms
SPATIAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SPECTRAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
