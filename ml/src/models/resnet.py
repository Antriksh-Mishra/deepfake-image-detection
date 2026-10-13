import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ResNet50Detector(nn.Module):
    """ResNet50 Transfer Learning Classifier for Deepfake Detection with offline fallback."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNet50Detector, self).__init__()
        
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load online ResNet50 weights ({e}). Falling back to randomly initialized weights.")
            self.backbone = resnet50(weights=None)
        
        # Save target layer reference for Grad-CAM
        self.target_layer = self.backbone.layer4
        
        # Replace FC classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM explainability."""
        return self.target_layer
