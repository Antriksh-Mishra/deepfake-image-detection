import torch.nn as nn
from typing import Dict, Any, Type

from ml.src.models.custom_cnn import CustomCNN
from ml.src.models.resnet import ResNet50Detector
from ml.src.models.efficientnet import EfficientNetDetector

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn": CustomCNN,
    "custom_cnn": CustomCNN,
    "resnet50": ResNet50Detector,
    "efficientnet": EfficientNetDetector,
    "efficientnet_b0": EfficientNetDetector,
}

def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.5) -> nn.Module:
    """
    Factory function instantiating model architectures by configuration string.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        supported = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model name '{model_name}'. Supported options: {supported}")

    model_cls = MODEL_REGISTRY[key]
    
    if issubclass(model_cls, CustomCNN):
        return model_cls(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        return model_cls(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
