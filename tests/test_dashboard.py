"""
Unit tests for the Model Performance & Evaluation Dashboard module.
"""

from app.performance_dashboard import load_evaluation_data


def test_load_evaluation_data():
    data = load_evaluation_data()
    assert "metrics" in data
    assert "history" in data
    assert "checkpoint" in data
    
    # Check metrics (honest non-leaked evaluation range — never hardcode historical metrics)
    metrics = data["metrics"]
    assert "accuracy" in metrics
    assert "auc_roc" in metrics
    assert 0.50 <= metrics["accuracy"] <= 1.0
    assert 0.50 <= metrics["auc_roc"] <= 1.0
    assert "test" in metrics
    assert "validation" in metrics
    assert metrics["sample_count"] == 400
    assert "calibration" in data
    
    # Check history
    history = data["history"]
    assert "train_loss" in history
    assert len(history["train_loss"]) == 6
    
    # Check checkpoint info
    ckpt = data["checkpoint"]
    assert ckpt["exists"] is True
    assert ckpt["sha256"] != "N/A"
    assert len(ckpt["sha256"]) == 64
