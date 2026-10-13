import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import logging

logger = logging.getLogger(__name__)

class EfficientNetDetector(nn.Module):
    """EfficientNet-B0 Transfer Learning Classifier for Deepfake Detection with offline fallback."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.5):
        super(EfficientNetDetector, self).__init__()
        
        try:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load online EfficientNet weights ({e}). Falling back to randomly initialized weights.")
            self.backbone = efficientnet_b0(weights=None)
        
        # Target layer for Grad-CAM explainability
        self.target_layer = self.backbone.features[-1]
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM explainability."""
        return self.target_layer
