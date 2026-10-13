import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

EXPLAINABILITY_DISCLAIMER = (
    "Grad-CAM visualizes region influence on model decision scores. "
    "Highlighted areas indicate features that contributed most to the prediction score "
    "and do not constitute definitive proof that a specific region was independently manipulated."
)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Visualizes regions in input images that strongly influence deepfake classification predictions.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Identify target layer if not explicitly specified
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer"):
            self.target_layer = model.get_target_layer()
        elif hasattr(model, "target_layer"):
            self.target_layer = model.target_layer
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            self.target_layer = model.backbone.layer4
        elif hasattr(model, "conv4"):
            self.target_layer = model.conv4
        else:
            # Fallback: search for last Conv2d layer in model modules
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if conv_layers:
                self.target_layer = conv_layers[-1]
            else:
                raise ValueError("Could not automatically locate a Conv2d target layer for Grad-CAM.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates Grad-CAM heatmap array normalized in [0, 1] matching input spatial dimensions (H, W).
        Input tensor shape: (1, C, H, W)
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Zero gradients & backward pass
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients during Grad-CAM execution.")

        # Compute neuron importance weights (Global Average Pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to keep positive contributions

        # Resize CAM to match input image spatial resolution (H, W)
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize CAM values to [0.0, 1.0]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.squeeze().cpu().numpy()
        return cam_np
