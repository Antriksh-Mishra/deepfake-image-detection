import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from ml.src.preprocessing.augmentations import get_transforms
from ml.src.preprocessing.pipeline import ImagePreprocessor

logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    0: "real",
    1: "deepfake"
}

REVERSE_CLASS_MAPPING = {
    "real": 0,
    "fake": 1,
    "deepfake": 1
}

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Detection.
    Scans directory layout:
        root_dir/
            real/
            fake/ (or deepfake/)
    """

    def __init__(self, root_dir: str, is_train: bool = True, image_size: int = 224, enable_face_crop: bool = False):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.image_size = image_size
        self.enable_face_crop = enable_face_crop
        self.transforms = get_transforms(is_train=is_train, image_size=image_size)
        self.preprocessor = ImagePreprocessor(image_size=image_size, enable_face_detection=enable_face_crop)
        
        self.samples: List[Tuple[Path, int]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Indexes all real and fake images in the specified split root directory."""
        if not self.root_dir.exists():
            logger.warning(f"Dataset directory '{self.root_dir}' does not exist.")
            return

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

        for class_name, label_id in REVERSE_CLASS_MAPPING.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists() and class_dir.is_dir():
                for file_path in class_dir.rglob("*"):
                    if file_path.suffix.lower() in valid_extensions:
                        self.samples.append((file_path, label_id))

        logger.info(f"Indexed {len(self.samples)} total samples from '{self.root_dir}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        try:
            if self.enable_face_crop:
                res = self.preprocessor.process(str(file_path), require_face=False)
                tensor = res["tensor"]
            else:
                img = Image.open(file_path).convert("RGB")
                tensor = self.transforms(img)
            return tensor, label
        except Exception as e:
            logger.error(f"Error loading image '{file_path}': {str(e)}")
            # Return dummy tensor in case of corrupted file during training loop
            return torch.zeros((3, self.image_size, self.image_size)), label

def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """Factory helper creating DataLoaders for train, validation, and test splits."""
    root = Path(dataset_dir)
    loaders = {}

    for split in ["train", "validation", "val", "test"]:
        split_path = root / split
        if split_path.exists():
            is_train = (split == "train")
            ds = DeepfakeDataset(str(split_path), is_train=is_train, image_size=image_size)
            if len(ds) > 0:
                loader_key = "val" if split in ["validation", "val"] else split
                loaders[loader_key] = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=is_train,
                    num_workers=num_workers,
                    pin_memory=True
                )

    return loaders
