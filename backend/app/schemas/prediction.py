from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class PredictionResponse(BaseModel):
    id: int
    image_name: str
    prediction: str
    confidence: float
    model: str
    model_version: str
    processing_time_ms: float
    original_image_url: Optional[str] = None
    processed_image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True

class PredictionHistoryItem(BaseModel):
    id: int
    image_name: str
    prediction: str
    confidence: float
    model: str
    processing_time_ms: float
    created_at: str

class PredictionHistoryResponse(BaseModel):
    items: List[PredictionResponse]
    total: int
    page: int
    size: int
    total_pages: int

class DashboardStatsResponse(BaseModel):
    total_analyses: int
    real_count: int
    deepfake_count: int
    avg_confidence: float
    recent_predictions: List[PredictionResponse]

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    image_size: int
    confidence_threshold: float
    supported_formats: List[str]
    max_upload_size_mb: int

class ApiErrorResponse(BaseModel):
    detail: str
    error_code: str
    status_code: int
