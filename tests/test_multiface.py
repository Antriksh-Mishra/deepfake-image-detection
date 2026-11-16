"""
Unit tests for Multiple Face Extraction and Multi-Face Pipeline Handling.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from src.core.face_extractor import FaceExtractor
from src.pipeline import ForensicInferencePipeline


def test_extract_all_faces():
    extractor = FaceExtractor(target_size=(256, 256))
    img = Image.new("RGB", (400, 300), color=(180, 150, 120))

    # Fallback center crop on featureless synthetic canvas
    all_faces = extractor.extract_all_faces(img)
    assert len(all_faces) >= 1
    assert "aligned_face" in all_faces[0]
    assert all_faces[0]["aligned_face"].size == (256, 256)
    assert "is_primary" in all_faces[0]
    assert all_faces[0]["is_primary"] is True


def test_face_extractor_rejects_non_skin_text_pattern():
    """Verifies B4 fix: high-contrast black-and-white text/patterns never pass skin+eye validation."""
    extractor = FaceExtractor(target_size=(256, 256))
    patch = np.zeros((140, 140, 3), dtype=np.uint8)
    cv2.putText(patch, "DEEPFAKE", (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(patch, "O = O", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    # Force vivid non-skin blue background so it isn't grayscale
    patch[:, :, 2] = 200
    assert extractor._is_valid_face_candidate(patch) is False


def test_pipeline_multiface_per_face_xai():
    """Verifies multi-face detection produces individual Grad-CAM, FFT, and ELA for every detected face."""
    p1 = Path("data/demo_samples/authentic_studio_photo_1.jpg")
    p2 = Path("data/demo_samples/deepfake_stylegan_checkerboard.jpg")

    pipe = ForensicInferencePipeline(device="cpu")
    if p1.exists() and p2.exists():
        f1 = Image.open(p1).convert("RGB").resize((240, 240))
        f2 = Image.open(p2).convert("RGB").resize((240, 240))
        canvas = Image.new("RGB", (680, 360), color=(35, 55, 80))
        canvas.paste(f1, (50, 60))
        canvas.paste(f2, (390, 60))

        res = pipe.analyze_image(canvas)
        assert res["faces_count"] == 2
        assert len(res["multi_face_reports"]) == 2
        assert res["full_image_annotated"].size == (680, 360)
        for rep in res["multi_face_reports"]:
            assert rep["gradcam_overlay"].size == (256, 256)
            assert rep["gradcam_heatmap"].shape == (256, 256)
            assert rep["fft_log_colormap_img"].size == (256, 256)
            assert rep["ela_heatmap_color"].size == (256, 256)
            assert "dominant_region" in rep["spatial_landmarks"]
    else:
        dummy = Image.new("RGB", (300, 300), color=(140, 100, 80))
        res = pipe.analyze_image(dummy)
        assert "faces_count" in res
        assert "multi_face_reports" in res
        assert isinstance(res["multi_face_reports"], list)
