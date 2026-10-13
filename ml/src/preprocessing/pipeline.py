import io
from PIL import Image, ImageOps
import torch
from typing import Dict, Any, Tuple, Optional, Union
import logging

from ml.src.preprocessing.face_detector import FaceDetector
from ml.src.preprocessing.augmentations import get_transforms
from ml.src.preprocessing.exceptions import (
    InvalidImageError,
    UnsupportedFormatError,
    FileTooLargeError,
    NoFaceDetectedError
)

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

class ImagePreprocessor:
    """End-to-End Image Preprocessing Pipeline for Deepfake Detection."""

    def __init__(self, image_size: int = 224, enable_face_detection: bool = True, max_size_bytes: int = MAX_FILE_SIZE_BYTES):
        self.image_size = image_size
        self.enable_face_detection = enable_face_detection
        self.max_size_bytes = max_size_bytes
        self.face_detector = FaceDetector()
        self.transform = get_transforms(is_train=False, image_size=image_size)

    def validate_raw_bytes(self, image_bytes: bytes) -> None:
        """Validates raw image binary size."""
        if len(image_bytes) > self.max_size_bytes:
            raise FileTooLargeError(f"Image file size ({len(image_bytes) / (1024*1024):.2f}MB) exceeds maximum limit of {self.max_size_bytes / (1024*1024):.1f}MB.")

    def process(self, image_input: Union[bytes, Image.Image, str], require_face: bool = True) -> Dict[str, Any]:
        """
        Executes full preprocessing pipeline:
        Input -> Validation -> Format Verification -> Face Crop -> Resize -> Normalization -> PyTorch Tensor.
        """
        # Load image into PIL
        if isinstance(image_input, bytes):
            self.validate_raw_bytes(image_input)
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                pil_img.verify()
                # Re-open after verify()
                pil_img = Image.open(io.BytesIO(image_input))
            except Exception as e:
                raise InvalidImageError(f"Failed to read image bytes: {str(e)}")
        elif isinstance(image_input, str):
            try:
                pil_img = Image.open(image_input)
            except Exception as e:
                raise InvalidImageError(f"Failed to open image from path '{image_input}': {str(e)}")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise InvalidImageError("Unsupported input type for image preprocessor.")

        # Check format
        img_format = (pil_img.format or "JPEG").upper()
        if img_format not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(f"Image format '{img_format}' is not supported. Allowed formats: {SUPPORTED_FORMATS}")

        # Correct EXIF orientation
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass

        pil_img = pil_img.convert("RGB")
        original_size = pil_img.size

        # Face Detection & Cropping
        face_bbox = None
        if self.enable_face_detection:
            processed_img, face_bbox = self.face_detector.detect_and_crop(pil_img, require_face=require_face)
        else:
            processed_img = pil_img

        # Transform to normalized PyTorch tensor
        tensor = self.transform(processed_img)

        return {
            "tensor": tensor,  # Shape: (3, 224, 224)
            "batch_tensor": tensor.unsqueeze(0),  # Shape: (1, 3, 224, 224)
            "original_image": pil_img,
            "processed_image": processed_img,
            "original_size": original_size,
            "face_bbox": face_bbox,
            "format": img_format
        }
