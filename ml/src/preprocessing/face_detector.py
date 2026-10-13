import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import logging
from ml.src.preprocessing.exceptions import NoFaceDetectedError

logger = logging.getLogger(__name__)

class FaceDetector:
    """OpenCV-based Face Detector and Cropper for deepfake image analysis."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60), margin_percent: float = 0.2):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margin_percent = margin_percent
        
        # Load OpenCV Haar Cascade for Frontal Face Detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning(f"Failed to load OpenCV Haar Cascade from {cascade_path}")

    def detect_and_crop(self, image: Image.Image, require_face: bool = True) -> Tuple[Image.Image, Optional[Dict[str, int]]]:
        """
        Detects primary face in PIL Image and returns cropped face image with bounding box metadata.
        If no face is detected and require_face is True, raises NoFaceDetectedError.
        If require_face is False, returns original image.
        """
        # Convert PIL to OpenCV BGR format
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        if len(faces) == 0:
            if require_face:
                raise NoFaceDetectedError("No human face detected in the provided image.")
            logger.info("No face detected; returning original image.")
            return image, None

        # Select the largest face by area
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Add margin/padding around detected face
        img_h, img_w, _ = img_np.shape
        margin_w = int(w * self.margin_percent)
        margin_h = int(h * self.margin_percent)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        crop_np = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop_np)

        bbox_meta = {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "crop_x1": int(x1),
            "crop_y1": int(y1),
            "crop_x2": int(x2),
            "crop_y2": int(y2)
        }

        return crop_pil, bbox_meta
