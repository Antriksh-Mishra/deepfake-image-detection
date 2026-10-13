import os
import uuid
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StorageService:
    """Service for managing file storage of uploaded images, face crops, heatmaps, and overlays."""

    def __init__(self, base_storage_dir: str = "storage"):
        self.base_dir = os.path.abspath(base_storage_dir)
        self.uploads_dir = os.path.join(self.base_dir, "uploads")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.heatmaps_dir = os.path.join(self.base_dir, "heatmaps")
        self.overlays_dir = os.path.join(self.base_dir, "overlays")

        for d in [self.uploads_dir, self.processed_dir, self.heatmaps_dir, self.overlays_dir]:
            os.makedirs(d, exist_ok=True)

    def save_image_bytes(self, image_bytes: bytes, filename: str) -> str:
        """Saves uploaded raw image bytes to uploads directory."""
        ext = os.path.splitext(filename)[1].lower() or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(self.uploads_dir, unique_name)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        return os.path.join("uploads", unique_name)

    def save_pil_image(self, image: Image.Image, category: str = "processed") -> str:
        """Saves PIL Image to specified storage subfolder (processed, heatmaps, overlays)."""
        target_dir = getattr(self, f"{category}_dir", self.processed_dir)
        unique_name = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(target_dir, unique_name)
        image.save(file_path, format="JPEG", quality=90)
        return os.path.join(category, unique_name)

    def delete_file(self, relative_path: Optional[str]) -> None:
        """Deletes stored file from disk if it exists."""
        if not relative_path:
            return
        abs_path = os.path.join(self.base_dir, relative_path)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except Exception as e:
                logger.error(f"Failed to delete file '{abs_path}': {str(e)}")

    def build_url(self, relative_path: Optional[str], base_url: str = "") -> Optional[str]:
        """Converts relative storage path to full web URL."""
        if not relative_path:
            return None
        clean_path = relative_path.replace("\\", "/").lstrip("/")
        return f"/storage/{clean_path}"
