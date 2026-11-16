"""
Spatial Stream Backbone.
Leverages pre-trained CNN backbones (EfficientNet-B0 or ResNet34)
to extract fine-grained spatial facial representations and artifacts.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet34, ResNet34_Weights


class SpatialBackbone(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True, feat_dim: int = 512):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "efficientnet_b0":
            try:
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base_model = efficientnet_b0(weights=weights)
            except Exception as e:
                print(f"[i] Could not download pretrained weights ({e}); initializing without pre-training.")
                base_model = efficientnet_b0(weights=None)
            
            # The convolutional feature extractor
            self.features = base_model.features
            # The final conv layer to hook for Grad-CAM
            self.target_cam_layer = self.features[-1]
            
            in_features = base_model.classifier[1].in_features  # 1280
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Linear(in_features, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True)
            )
        elif backbone_name == "resnet34":
            try:
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                base_model = resnet34(weights=weights)
            except Exception as e:
                print(f"[i] Could not download pretrained weights ({e}); initializing without pre-training.")
                base_model = resnet34(weights=None)
            
            self.features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            self.target_cam_layer = base_model.layer4[-1]
            in_features = base_model.fc.in_features  # 512
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Linear(in_features, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized RGB image.
            
        Returns:
            spatial_feat: (B, feat_dim) feature vector.
        """
        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        flat = torch.flatten(pooled, 1)
        spatial_feat = self.proj(flat)
        return spatial_feat
