"""
Frequency Stream & Steganalysis Feature Extractor.
Contains the Spatial Rich Model (SRM) high-pass filter bank and a dedicated
spectral ConvNet encoder to extract forensic noise residuals and 2D FFT anomalies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_srm_filters() -> torch.Tensor:
    """
    Returns 3 standard SRM (Spatial Rich Model) high-pass filters used in digital forensics:
    Filter 1: 1st-order directional residual
    Filter 2: 2nd-order Laplacian-type residual
    Filter 3: 3x3 square edge residual
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

    # Stack as (3, 1, 5, 5) filter tensor
    filters = np.stack([f1, f2, f3], axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(filters)


class SRMConv(nn.Module):
    """Fixed Spatial Rich Model high-pass convolution layer."""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        filters = get_srm_filters()
        self.register_buffer("weight", filters)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        return F.conv2d(x, self.weight, padding=2)


class FrequencyStream(nn.Module):
    """
    Extracts 256-dimensional forensic frequency representations
    from the 2D FFT magnitude spectrum and high-pass SRM residuals.
    """
    def __init__(self, in_channels: int = 1, feat_dim: int = 256):
        super().__init__()
        self.srm = SRMConv(in_channels=1)
        
        # 4-stage convolutional encoder
        # Inputs: original spectral map (1 ch) + 3 SRM filter residuals = 4 channels
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, spectral_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral_tensor: (B, 1, H, W) normalized 2D FFT magnitude spectrum.
            
        Returns:
            spectral_feat: (B, feat_dim) feature vector.
        """
        # Compute SRM residuals on the spectral tensor
        residuals = self.srm(spectral_tensor)  # (B, 3, H, W)
        
        # Concatenate original spectrum and 3 SRM filtered channels -> (B, 4, H, W)
        combined = torch.cat([spectral_tensor, residuals], dim=1)
        
        x = self.conv_net(combined)
        x = torch.flatten(x, 1)
        spectral_feat = self.fc(x)
        return spectral_feat
