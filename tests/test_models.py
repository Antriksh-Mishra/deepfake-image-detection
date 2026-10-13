import unittest
import torch
import torch.nn as nn
from ml.src.models.custom_cnn import CustomCNN
from ml.src.models.resnet import ResNet50Detector
from ml.src.models.efficientnet import EfficientNetDetector
from ml.src.models.factory import get_model, MODEL_REGISTRY
from ml.src.evaluation.evaluator import ModelEvaluator

class TestModelArchitectures(unittest.TestCase):

    def setUp(self):
        # Dummy batch of 2 images: (B=2, C=3, H=224, W=224)
        self.dummy_input = torch.randn(2, 3, 224, 224)

    def test_custom_cnn_output_shape(self):
        model = CustomCNN(num_classes=2)
        model.eval()
        with torch.no_grad():
            output = model(self.dummy_input)
        self.assertEqual(output.shape, (2, 2))

    def test_resnet50_output_shape(self):
        model = ResNet50Detector(num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(self.dummy_input)
        self.assertEqual(output.shape, (2, 2))

    def test_efficientnet_output_shape(self):
        model = EfficientNetDetector(num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(self.dummy_input)
        self.assertEqual(output.shape, (2, 2))

    def test_factory_instantiation(self):
        cnn = get_model("cnn", num_classes=2, pretrained=False)
        resnet = get_model("resnet50", num_classes=2, pretrained=False)
        eff = get_model("efficientnet_b0", num_classes=2, pretrained=False)

        self.assertIsInstance(cnn, CustomCNN)
        self.assertIsInstance(resnet, ResNet50Detector)
        self.assertIsInstance(eff, EfficientNetDetector)

    def test_factory_invalid_name(self):
        with self.assertRaises(ValueError):
            get_model("invalid_model_xyz")

    def test_evaluator_metrics(self):
        model = CustomCNN(num_classes=2)
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create dummy DataLoader containing 4 samples
        dummy_data = [(torch.randn(3, 224, 224), 0) for _ in range(2)] + [(torch.randn(3, 224, 224), 1) for _ in range(2)]
        loader = torch.utils.data.DataLoader(dummy_data, batch_size=2)
        
        metrics = evaluator.evaluate(loader)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertEqual(metrics["total_samples"], 4)

if __name__ == "__main__":
    unittest.main()
