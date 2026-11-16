"""
Unit tests for End-to-End ForensicInferencePipeline.
"""

import pytest
from PIL import Image
from src.pipeline import ForensicInferencePipeline


def test_pipeline_inference():
    pipeline = ForensicInferencePipeline(device="cpu")
    # Test on arbitrary RGB image
    dummy_img = Image.new("RGB", (300, 300), color=(150, 120, 100))
    
    results = pipeline.analyze_image(dummy_img)
    
    assert "prediction_label" in results
    assert "is_fake" in results
    assert "fake_probability" in results
    assert "real_probability" in results
    assert "gradcam_heatmap" in results
    assert "fft_spectrum_img" in results
    assert "ela_image" in results
    assert "verdict_tier" in results
    assert "latency_breakdown" in results
    assert "total_ms" in results["latency_breakdown"]
    assert 0.0 <= results["fake_probability"] <= 1.0
    assert results["aligned_face"].size == (256, 256)
