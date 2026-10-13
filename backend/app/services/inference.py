import time
import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Optional
import logging

from app.config import settings
from app.services.storage import StorageService
from ml.src.preprocessing.pipeline import ImagePreprocessor
from ml.src.models.factory import get_model
from ml.src.explainability.gradcam import GradCAM
from ml.src.explainability.visualizer import ExplainabilityVisualizer
from ml.src.datasets.dataset import CLASS_MAPPING

logger = logging.getLogger(__name__)

class InferenceService:
    """Managed Inference Service executing end-to-end model prediction and Grad-CAM generation."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Inference Service on device: {self.device}")

        self.preprocessor = ImagePreprocessor(image_size=settings.IMAGE_SIZE, enable_face_detection=True)
        self.storage = StorageService()

        # Load PyTorch model
        self.model_name = settings.MODEL_NAME
        self.model_version = settings.MODEL_VERSION
        self.model = get_model(self.model_name, num_classes=2, pretrained=True)

        model_path = settings.MODEL_PATH
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded model weights from '{model_path}'.")
            except Exception as e:
                logger.warning(f"Could not load checkpoint '{model_path}': {str(e)}. Using reference weights.")
        else:
            logger.info(f"No checkpoint found at '{model_path}'. Initializing reference weights.")

        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM and Visualizer
        self.gradcam = GradCAM(self.model)
        self.visualizer = ExplainabilityVisualizer(alpha=0.5)

    def predict(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Executes complete prediction pipeline:
        Preprocessing -> Face Crop -> Model Forward Pass -> Softmax Probabilities -> Grad-CAM Heatmap -> Storage.
        """
        t0 = time.time()

        # Save raw original upload
        orig_rel_path = self.storage.save_image_bytes(image_bytes, filename)

        # Preprocess & face crop
        res = self.preprocessor.process(image_bytes, require_face=True)
        input_tensor = res["batch_tensor"].to(self.device)
        processed_crop = res["processed_image"]

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_class_idx = int(probs.argmax().item())
            confidence = float(probs[pred_class_idx].item())

        # Determine label: Class 1 = DEEPFAKE, Class 0 = REAL
        label = "DEEPFAKE" if pred_class_idx == 1 else "REAL"

        # Generate Grad-CAM explainability artifacts
        explain_res = self.visualizer.generate(
            gradcam_engine=self.gradcam,
            input_tensor=input_tensor,
            image_crop=processed_crop,
            target_class=pred_class_idx
        )

        # Save media output artifacts
        proc_rel_path = self.storage.save_pil_image(processed_crop, category="processed")
        heatmap_rel_path = self.storage.save_pil_image(explain_res["heatmap_image"], category="heatmaps")
        overlay_rel_path = self.storage.save_pil_image(explain_res["overlay_image"], category="overlays")

        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "image_name": filename,
            "prediction": label,
            "confidence": round(confidence, 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_time_ms": latency_ms,
            "original_image_path": orig_rel_path,
            "processed_image_path": proc_rel_path,
            "heatmap_path": heatmap_rel_path,
            "overlay_path": overlay_rel_path,
            "original_image_url": self.storage.build_url(orig_rel_path),
            "processed_image_url": self.storage.build_url(proc_rel_path),
            "heatmap_url": self.storage.build_url(heatmap_rel_path),
            "overlay_url": self.storage.build_url(overlay_rel_path),
            "disclaimer": explain_res["disclaimer"]
        }

# Global Inference Service instance
_inference_service_instance: Optional[InferenceService] = None

def get_inference_service() -> InferenceService:
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService()
    return _inference_service_instance
