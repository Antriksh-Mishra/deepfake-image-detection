import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from ml.src.models.custom_cnn import CustomCNN
from ml.src.models.resnet import ResNet50Detector
from ml.src.explainability.gradcam import GradCAM, EXPLAINABILITY_DISCLAIMER
from ml.src.explainability.visualizer import ExplainabilityVisualizer

class TestGradCAMExplainability(unittest.TestCase):

    def setUp(self):
        self.model = CustomCNN(num_classes=2)
        self.model.eval()
        self.dummy_tensor = torch.randn(1, 3, 224, 224)
        self.dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    def test_gradcam_generation(self):
        gradcam = GradCAM(self.model)
        cam = gradcam.generate(self.dummy_tensor, target_class=1)
        
        self.assertEqual(cam.shape, (224, 224))
        self.assertTrue(np.all(cam >= 0.0) and np.all(cam <= 1.0))

    def test_explainability_visualizer(self):
        gradcam = GradCAM(self.model)
        visualizer = ExplainabilityVisualizer(alpha=0.5)
        
        result = visualizer.generate(gradcam, self.dummy_tensor, self.dummy_image, target_class=1)
        
        self.assertIn("cam", result)
        self.assertIn("heatmap_image", result)
        self.assertIn("overlay_image", result)
        self.assertIn("disclaimer", result)
        
        self.assertIsInstance(result["heatmap_image"], Image.Image)
        self.assertIsInstance(result["overlay_image"], Image.Image)
        self.assertEqual(result["heatmap_image"].size, (224, 224))
        self.assertEqual(result["overlay_image"].size, (224, 224))
        self.assertEqual(result["disclaimer"], EXPLAINABILITY_DISCLAIMER)

if __name__ == "__main__":
    unittest.main()
