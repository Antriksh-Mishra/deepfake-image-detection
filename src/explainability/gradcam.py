"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.
Provides visual explanations of CNN decisions by computing gradient flows
with respect to the final convolutional feature maps, highlighting spatial
anomalies (e.g., blending boundaries, distorted facial features).
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_regional_scores(
    cam_heatmap: np.ndarray,
    has_face: bool = True,
    image_type: Optional[str] = None,
) -> Dict[str, object]:
    """
    Summarizes normalized Grad-CAM activation across spatial zones.
    If a face is present, zones correspond to facial anatomy (ocular, mid-facial, jawline).
    If no face is present (e.g. shoes, vehicles, products, artwork, animals), zones correspond
    to domain-appropriate object geometry (upper structure, central core, lower base).
    """
    h_cam = cam_heatmap.shape[0]
    upper_score = float(np.mean(cam_heatmap[:max(1, int(h_cam * 0.40)), :]))
    mid_score = float(np.mean(cam_heatmap[int(h_cam * 0.40):max(1, int(h_cam * 0.75)), :]))
    lower_score = float(np.mean(cam_heatmap[int(h_cam * 0.75):, :]))

    if has_face:
        region_scores = {
            "Ocular / Eyes Region": upper_score,
            "Mid-Facial / Nasolabial Region": mid_score,
            "Jawline / Boundary Seam": lower_score,
        }
        dominant_region = max(region_scores, key=region_scores.get)
        return {
            "region_scores": region_scores,
            "dominant_region": dominant_region,
            "ocular_score": upper_score,
            "mid_facial_score": mid_score,
            "boundary_score": lower_score,
            "is_facial_context": True,
            "context_type": "Human Face Portrait",
        }
    else:
        # Object-aware spatial zone descriptors for non-face media
        ctx = image_type or "General Object"
        region_scores = {
            "Upper Structural Perimeter": upper_score,
            "Central Object Core / Foreground": mid_score,
            "Lower Base / Support Boundary": lower_score,
        }
        dominant_region = max(region_scores, key=region_scores.get)
        return {
            "region_scores": region_scores,
            "dominant_region": dominant_region,
            "upper_object_score": upper_score,
            "central_core_score": mid_score,
            "lower_base_score": lower_score,
            # Backward compatibility aliases for existing telemetry readers
            "ocular_score": upper_score,
            "mid_facial_score": mid_score,
            "boundary_score": lower_score,
            "is_facial_context": False,
            "context_type": ctx,
        }


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch classification model.
            target_layer: The convolutional layer to hook for activation maps.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        spatial_tensor: torch.Tensor,
        spectral_tensor: Optional[torch.Tensor] = None,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for the input tensor.

        Args:
            spatial_tensor: (1, 3, H, W) input image tensor.
            spectral_tensor: Optional (1, 1, H, W) frequency input tensor.
            target_class: Target class index (1 = Manipulated, 0 = Authentic).
                          For single-logit binary outputs, target_class=0 maximizes (-logit).

        Returns:
            cam_heatmap: 2D np.ndarray float32 normalized in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        self.activations = None

        if not spatial_tensor.requires_grad and not any(p.requires_grad for p in self.model.parameters()):
            spatial_tensor = spatial_tensor.detach().requires_grad_(True)

        # Forward pass
        if spectral_tensor is not None:
            output = self.model(spatial_tensor, spectral_tensor)
        else:
            output = self.model(spatial_tensor)

        if output.dim() == 1:
            raw_logit = output[0]
            score = raw_logit if int(target_class) == 1 else -raw_logit
        elif output.size(1) == 1:
            raw_logit = output[0, 0]
            score = raw_logit if int(target_class) == 1 else -raw_logit
        else:
            cls_idx = int(np.clip(target_class, 0, output.size(1) - 1))
            score = output[0, cls_idx]

        # Backward pass (no retain_graph leak during inference)
        score.backward()
        self.model.zero_grad(set_to_none=True)

        h_in, w_in = spatial_tensor.shape[2], spatial_tensor.shape[3]
        if self.gradients is None or self.activations is None:
            return np.zeros((h_in, w_in), dtype=np.float32)

        gradients = self.gradients[0]      # (C, H_feat, W_feat)
        activations = self.activations[0]  # (C, H_feat, W_feat)

        # Global Average Pooling of gradients to compute importance weights
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        cam_min, cam_max = float(np.min(cam_np)), float(np.max(cam_np))
        if cam_max - cam_min > 1e-7:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        cam_resized = cv2.resize(cam_np, (w_in, h_in), interpolation=cv2.INTER_CUBIC)
        return np.clip(cam_resized, 0.0, 1.0).astype(np.float32)

    def overlay_cam(
        self,
        original_img_np: np.ndarray,
        cam_heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays the CAM heatmap safely onto the original face RGB image.

        Verifies and reconciles spatial shape mismatches before blending.
        """
        img_arr = np.asarray(original_img_np)
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[2] == 4:
            img_arr = img_arr[:, :, :3]

        if img_arr.dtype != np.uint8:
            max_val = float(np.max(img_arr)) if img_arr.size else 1.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            img_arr = np.clip(img_arr * scale, 0, 255).astype(np.uint8)

        h_img, w_img = img_arr.shape[:2]
        cam = np.asarray(cam_heatmap, dtype=np.float32)
        if cam.shape[:2] != (h_img, w_img):
            cam = cv2.resize(cam, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        cam = np.clip(cam, 0.0, 1.0)

        heatmap_uint8 = (cam * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = (alpha * heatmap_color.astype(np.float32) + (1.0 - alpha) * img_arr.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
