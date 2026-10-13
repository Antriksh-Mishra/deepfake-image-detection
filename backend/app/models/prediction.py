from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.models.database import Base

class Prediction(Base):
    """Prediction history database model schema."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_name = Column(String(255), nullable=False, index=True)
    prediction = Column(String(50), nullable=False, index=True)  # "REAL" or "DEEPFAKE"
    confidence = Column(Float, nullable=False)                    # e.g., 0.9632
    model_name = Column(String(100), nullable=False)              # e.g., "resnet50"
    model_version = Column(String(50), nullable=False, default="1.0.0")
    processing_time_ms = Column(Float, nullable=False)            # Latency in ms
    
    # Relative file storage paths
    original_image_path = Column(String(500), nullable=True)
    processed_image_path = Column(String(500), nullable=True)
    heatmap_path = Column(String(500), nullable=True)
    overlay_path = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
