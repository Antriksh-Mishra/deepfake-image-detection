import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import logging
from ml.src.preprocessing.exceptions import NoFaceDetectedError

logger = logging.getLogger(__name__)

class FaceDetector:
    """OpenCV-based Face Detector and Cropper with resilient fallback handling."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60), margin_percent: float = 0.2):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margin_percent = margin_percent
        self.face_cascade = None

        # Safely attempt to load OpenCV Haar Cascade if available in this build
        if hasattr(cv2, "CascadeClassifier"):
            cascade_path = None
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            
            if cascade_path and os.path.exists(cascade_path):
                try:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.face_cascade = cascade
                        logger.info("Initialized OpenCV Haar Cascade face detector.")
                except Exception as e:
                    logger.warning(f"Failed to load OpenCV Haar Cascade: {e}")

        if self.face_cascade is None:
            logger.info("OpenCV CascadeClassifier not available in this environment; center face crop fallback enabled.")

    def detect_and_crop(self, image: Image.Image, require_face: bool = True) -> Tuple[Image.Image, Optional[Dict[str, int]]]:
        """
        Detects primary face in PIL Image and returns cropped face image with bounding box metadata.
        Falls back to centered facial framing if Haar cascade is absent in the installed OpenCV version.
        """
        img_np = np.array(image.convert("RGB"))
        img_h, img_w, _ = img_np.shape

        if self.face_cascade is not None:
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

            # Select largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

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

        else:
            # Fallback: portrait-centric cropping
            crop_size = min(img_w, img_h)
            x1 = max(0, (img_w - crop_size) // 2)
            y1 = max(0, int((img_h - crop_size) * 0.2))
            x2 = min(img_w, x1 + crop_size)
            y2 = min(img_h, y1 + crop_size)

            crop_np = img_np[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop_np)

            bbox_meta = {
                "x": int(x1),
                "y": int(y1),
                "width": int(crop_size),
                "height": int(crop_size),
                "crop_x1": int(x1),
                "crop_y1": int(y1),
                "crop_x2": int(x2),
                "crop_y2": int(y2)
            }
            return crop_pil, bbox_meta
