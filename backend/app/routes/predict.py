from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
import math

from app.models.database import get_db
from app.models.prediction import Prediction
from app.schemas.prediction import (
    PredictionResponse,
    PredictionHistoryResponse,
    DashboardStatsResponse,
    ModelInfoResponse
)
from app.services.inference import get_inference_service, InferenceService
from app.services.storage import StorageService
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["Deepfake Detection"])
storage = StorageService()

@router.get("/model", response_model=ModelInfoResponse)
def get_model_info():
    """Returns currently deployed model information and technical parameters."""
    return ModelInfoResponse(
        model_name=settings.MODEL_NAME,
        model_version=settings.MODEL_VERSION,
        image_size=settings.IMAGE_SIZE,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        supported_formats=settings.ALLOWED_EXTENSIONS,
        max_upload_size_mb=settings.MAX_UPLOAD_SIZE_MB
    )

@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    inference: InferenceService = Depends(get_inference_service)
):
    """
    Main Deepfake Prediction Endpoint.
    Accepts multipart image file upload, extracts face, runs classification model,
    generates Grad-CAM visual heatmap overlay, and stores prediction history record.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    # Read binary bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run inference and Grad-CAM generation
    res = inference.predict(image_bytes=contents, filename=file.filename)

    # Persist prediction history to Database
    db_record = Prediction(
        image_name=res["image_name"],
        prediction=res["prediction"],
        confidence=res["confidence"],
        model_name=res["model_name"],
        model_version=res["model_version"],
        processing_time_ms=res["processing_time_ms"],
        original_image_path=res["original_image_path"],
        processed_image_path=res["processed_image_path"],
        heatmap_path=res["heatmap_path"],
        overlay_path=res["overlay_path"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return PredictionResponse(
        id=db_record.id,
        image_name=db_record.image_name,
        prediction=db_record.prediction,
        confidence=db_record.confidence,
        model=db_record.model_name,
        model_version=db_record.model_version,
        processing_time_ms=db_record.processing_time_ms,
        original_image_url=storage.build_url(db_record.original_image_path),
        processed_image_url=storage.build_url(db_record.processed_image_path),
        heatmap_url=storage.build_url(db_record.heatmap_path),
        overlay_url=storage.build_url(db_record.overlay_path),
        created_at=db_record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    )

@router.get("/predictions", response_model=PredictionHistoryResponse)
def get_prediction_history(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    filter_by: Optional[str] = Query(None, alias="filter"), # "REAL" or "DEEPFAKE"
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Returns paginated, searchable, and filterable prediction history records."""
    query = db.query(Prediction)

    if filter_by:
        query = query.filter(Prediction.prediction == filter_by.upper())

    if search:
        query = query.filter(Prediction.image_name.ilike(f"%{search}%"))

    total = query.count()
    total_pages = math.ceil(total / size) if total > 0 else 1

    records = query.order_by(Prediction.created_at.desc()).offset((page - 1) * size).limit(size).all()

    items = [
        PredictionResponse(
            id=r.id,
            image_name=r.image_name,
            prediction=r.prediction,
            confidence=r.confidence,
            model=r.model_name,
            model_version=r.model_version,
            processing_time_ms=r.processing_time_ms,
            original_image_url=storage.build_url(r.original_image_path),
            processed_image_url=storage.build_url(r.processed_image_path),
            heatmap_url=storage.build_url(r.heatmap_path),
            overlay_url=storage.build_url(r.overlay_path),
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )
        for r in records
    ]

    return PredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        total_pages=total_pages
    )

@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    """Retrieves single prediction detail by ID."""
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")

    return PredictionResponse(
        id=record.id,
        image_name=record.image_name,
        prediction=record.prediction,
        confidence=record.confidence,
        model=record.model_name,
        model_version=record.model_version,
        processing_time_ms=record.processing_time_ms,
        original_image_url=storage.build_url(record.original_image_path),
        processed_image_url=storage.build_url(record.processed_image_path),
        heatmap_url=storage.build_url(record.heatmap_path),
        overlay_url=storage.build_url(record.overlay_path),
        created_at=record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    )

@router.delete("/predictions/{prediction_id}", status_code=status.HTTP_200_OK)
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Deletes prediction record from DB and deletes associated image files from disk."""
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")

    # Delete disk files
    storage.delete_file(record.original_image_path)
    storage.delete_file(record.processed_image_path)
    storage.delete_file(record.heatmap_path)
    storage.delete_file(record.overlay_path)

    db.delete(record)
    db.commit()

    return {"message": f"Prediction #{prediction_id} deleted successfully."}

@router.get("/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Returns aggregated dashboard statistics computed from real database data."""
    total_analyses = db.query(Prediction).count()
    real_count = db.query(Prediction).filter(Prediction.prediction == "REAL").count()
    deepfake_count = db.query(Prediction).filter(Prediction.prediction == "DEEPFAKE").count()

    avg_conf_res = db.query(func.avg(Prediction.confidence)).scalar()
    avg_confidence = round(float(avg_conf_res or 0.0), 4)

    recent_records = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(5).all()

    recent_items = [
        PredictionResponse(
            id=r.id,
            image_name=r.image_name,
            prediction=r.prediction,
            confidence=r.confidence,
            model=r.model_name,
            model_version=r.model_version,
            processing_time_ms=r.processing_time_ms,
            original_image_url=storage.build_url(r.original_image_path),
            processed_image_url=storage.build_url(r.processed_image_path),
            heatmap_url=storage.build_url(r.heatmap_path),
            overlay_url=storage.build_url(r.overlay_path),
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )
        for r in recent_records
    ]

    return DashboardStatsResponse(
        total_analyses=total_analyses,
        real_count=real_count,
        deepfake_count=deepfake_count,
        avg_confidence=avg_confidence,
        recent_predictions=recent_items
    )
