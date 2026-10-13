import unittest
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml.src.preprocessing.pipeline import ImagePreprocessor
from ml.src.models.factory import get_model
from ml.src.explainability.gradcam import GradCAM
from ml.src.explainability.visualizer import ExplainabilityVisualizer
from backend.app.models.database import Base
from backend.app.models.prediction import Prediction
from backend.app.services.storage import StorageService

class TestEndToEndPipeline(unittest.TestCase):
    """End-to-End system verification test testing full pipeline flow."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = StorageService(base_storage_dir=os.path.join(self.temp_dir, "storage"))

        # Setup temporary SQLite database
        self.db_path = os.path.join(self.temp_dir, "e2e.db")
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create synthetic image bytes
        self.test_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        self.preprocessor = ImagePreprocessor(image_size=224, enable_face_detection=False)

        # Load PyTorch model
        self.model = get_model("resnet50", num_classes=2, pretrained=False)
        self.model.eval()

        self.gradcam = GradCAM(self.model)
        self.visualizer = ExplainabilityVisualizer(alpha=0.5)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline_flow(self):
        """Tests: Input -> Preprocessing -> Inference -> Grad-CAM -> Storage -> DB Log -> DB Query."""
        # 1. Preprocess image
        res = self.preprocessor.process(self.test_img, require_face=False)
        input_tensor = res["batch_tensor"]
        processed_crop = res["processed_image"]

        self.assertEqual(input_tensor.shape, (1, 3, 224, 224))

        # 2. Model Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())

        prediction_label = "DEEPFAKE" if pred_idx == 1 else "REAL"

        # 3. Grad-CAM Explainability
        explain_res = self.visualizer.generate(
            gradcam_engine=self.gradcam,
            input_tensor=input_tensor,
            image_crop=processed_crop,
            target_class=pred_idx
        )

        self.assertIn("heatmap_image", explain_res)
        self.assertIn("overlay_image", explain_res)

        # 4. Save media artifacts
        proc_path = self.storage.save_pil_image(processed_crop, category="processed")
        heatmap_path = self.storage.save_pil_image(explain_res["heatmap_image"], category="heatmaps")
        overlay_path = self.storage.save_pil_image(explain_res["overlay_image"], category="overlays")

        # 5. Persist to DB
        db = self.Session()
        record = Prediction(
            image_name="test_sample.jpg",
            prediction=prediction_label,
            confidence=round(confidence, 4),
            model_name="resnet50",
            model_version="1.0.0",
            processing_time_ms=124.5,
            original_image_path="uploads/test.jpg",
            processed_image_path=proc_path,
            heatmap_path=heatmap_path,
            overlay_path=overlay_path
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        # 6. Verify DB record retrieval
        fetched = db.query(Prediction).filter(Prediction.id == record.id).first()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.image_name, "test_sample.jpg")
        self.assertEqual(fetched.prediction, prediction_label)
        self.assertEqual(fetched.model_name, "resnet50")
        db.close()

if __name__ == "__main__":
    unittest.main()
