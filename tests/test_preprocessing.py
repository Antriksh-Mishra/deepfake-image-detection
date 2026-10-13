import unittest
import numpy as np
from PIL import Image
import torch
import tempfile
import os
import shutil

from ml.src.preprocessing.face_detector import FaceDetector
from ml.src.preprocessing.pipeline import ImagePreprocessor
from ml.src.preprocessing.exceptions import (
    InvalidImageError,
    UnsupportedFormatError,
    FileTooLargeError,
    NoFaceDetectedError
)
from ml.src.datasets.dataset import DeepfakeDataset

class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        # Create synthetic test image (224x224 RGB)
        self.test_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        self.preprocessor = ImagePreprocessor(image_size=224, enable_face_detection=False)

    def test_preprocessor_success(self):
        result = self.preprocessor.process(self.test_img, require_face=False)
        self.assertIn("tensor", result)
        self.assertIn("batch_tensor", result)
        self.assertEqual(result["tensor"].shape, (3, 224, 224))
        self.assertEqual(result["batch_tensor"].shape, (1, 3, 224, 224))

    def test_unsupported_format_exception(self):
        img_gif = Image.new("P", (100, 100))
        img_gif.format = "GIF"
        with self.assertRaises(UnsupportedFormatError):
            self.preprocessor.process(img_gif, require_face=False)

    def test_file_too_large_exception(self):
        large_bytes = b"0" * (11 * 1024 * 1024)  # 11MB
        with self.assertRaises(FileTooLargeError):
            self.preprocessor.process(large_bytes)

    def test_face_detector_no_face_exception(self):
        detector = FaceDetector()
        # Synthetic random noise image contains no human face
        with self.assertRaises(NoFaceDetectedError):
            detector.detect_and_crop(self.test_img, require_face=True)

class TestDeepfakeDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.temp_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "fake"), exist_ok=True)

        # Create sample images
        img1 = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img2 = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img1.save(os.path.join(self.temp_dir, "real", "sample1.jpg"))
        img2.save(os.path.join(self.temp_dir, "fake", "sample2.jpg"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_indexing(self):
        ds = DeepfakeDataset(self.temp_dir, is_train=True, image_size=224, enable_face_crop=False)
        self.assertEqual(len(ds), 2)
        tensor, label = ds[0]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])

if __name__ == "__main__":
    unittest.main()
