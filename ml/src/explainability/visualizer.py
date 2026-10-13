import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import logging

from ml.src.explainability.gradcam import GradCAM, EXPLAINABILITY_DISCLAIMER

logger = logging.getLogger(__name__)

class ExplainabilityVisualizer:
    """Service generating visual heatmaps and blended Grad-CAM overlays."""

    def __init__(self, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET):
        self.alpha = alpha
        self.colormap = colormap

    def overlay_heatmap(self, image: Image.Image, cam: np.ndarray) -> Tuple[Image.Image, Image.Image]:
        """
        Overlays a 2D numpy CAM array [0.0, 1.0] onto a PIL Image.
        Returns Tuple[heatmap_pil, overlay_pil].
        """
        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        h, w, _ = img_np.shape

        # Resize CAM to match image dimensions if needed
        if cam.shape != (h, w):
            cam = cv2.resize(cam, (w, h))

        # Convert CAM [0, 1] to uint8 heatmap
        heatmap_uint8 = np.uint8(255 * cam)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, self.colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend original image with heatmap
        overlay_np = cv2.addWeighted(img_np, 1.0 - self.alpha, heatmap_rgb, self.alpha, 0)

        heatmap_pil = Image.fromarray(heatmap_rgb)
        overlay_pil = Image.fromarray(overlay_np)

        return heatmap_pil, overlay_pil

    def generate(
        self,
        gradcam_engine: GradCAM,
        input_tensor: Any,
        image_crop: Image.Image,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates full explainability bundle including CAM array, Heatmap PIL image, Overlay PIL image, and Disclaimer.
        """
        cam_np = gradcam_engine.generate(input_tensor, target_class=target_class)
        heatmap_pil, overlay_pil = self.overlay_heatmap(image_crop, cam_np)

        return {
            "cam": cam_np,
            "heatmap_image": heatmap_pil,
            "overlay_image": overlay_pil,
            "disclaimer": EXPLAINABILITY_DISCLAIMER
        }
