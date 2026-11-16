"""
Loss functions for robust deepfake classification.
Includes Binary Cross-Entropy with Label Smoothing and Focal Loss
to mitigate overconfidence and address class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Label Smoothing.
    Smooths target labels from {0, 1} to [smoothing, 1 - smoothing]
    to prevent overconfident predictions on unseen generator artifacts.
    """
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: shape (B,) or (B, 1) float
        targets = targets.view_as(logits)
        smoothed_targets = targets * (1.0 - 2 * self.smoothing) + self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets)


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss: down-weights easy examples to focus gradient updates
    on hard-to-detect manipulated boundaries and subtle diffusion artifacts.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.50, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits).float()
        probs = torch.sigmoid(logits)
        
        # p_t is the probability corresponding to the true label
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        
        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        bce = -torch.log(p_t)
        loss = alpha_t * focal_weight * bce
        return loss.mean()
