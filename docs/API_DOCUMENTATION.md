# 📡 Deepfake Detection Platform - REST API Reference

The platform provides a standard REST API documented below with OpenAPI 3.0 specification available at `/api/v1/openapi.json`.

---

## 1. Health & System Status

### `GET /api/v1/health`
Checks server health and active model status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Deepfake Image Detection API",
  "version": "1.0.0",
  "model": "resnet50"
}
```

---

## 2. Model Information

### `GET /api/v1/model`
Returns currently deployed deep learning model metadata and technical parameters.

**Response (200 OK):**
```json
{
  "model_name": "resnet50",
  "model_version": "1.0.0",
  "image_size": 224,
  "confidence_threshold": 0.5,
  "supported_formats": ["jpg", "jpeg", "png", "webp"],
  "max_upload_size_mb": 10
}
```

---

## 3. Predict Deepfake

### `POST /api/v1/predict`
Upload an image to detect deepfake manipulations, extract face crop, run classification, generate Grad-CAM explainability artifacts, and store prediction history.

**Content-Type:** `multipart/form-data`  
**Parameters:**
- `file` (binary, required): Image file (JPEG, PNG, WEBP). Max size: 10 MB.

**Response (201 Created):**
```json
{
  "id": 142,
  "image_name": "profile_sample.jpg",
  "prediction": "DEEPFAKE",
  "confidence": 0.9632,
  "model": "resnet50",
  "model_version": "1.0.0",
  "processing_time_ms": 782.4,
  "original_image_url": "/storage/uploads/a1b2c3d4.jpg",
  "processed_image_url": "/storage/processed/e5f6g7h8.jpg",
  "heatmap_url": "/storage/heatmaps/i9j0k1l2.jpg",
  "overlay_url": "/storage/overlays/m3n4o5p6.jpg",
  "created_at": "2026-10-10 16:40:12"
}
```

**Error Responses:**
- `400 Bad Request` (`NO_FACE_DETECTED`): No human face detected in the image.
- `413 Payload Too Large` (`FILE_TOO_LARGE`): File size exceeds 10 MB limit.
- `415 Unsupported Media Type` (`UNSUPPORTED_MEDIA_TYPE`): Non-image file format.

---

## 4. Prediction History

### `GET /api/v1/predictions`
Retrieve paginated, searchable, and filterable past prediction records.

**Query Parameters:**
- `page` (int, default: 1): Page number.
- `size` (int, default: 10): Items per page (max: 100).
- `filter` (string, optional): `"REAL"` or `"DEEPFAKE"`.
- `search` (string, optional): Search query matching image name.

**Response (200 OK):**
```json
{
  "items": [
    {
      "id": 142,
      "image_name": "profile_sample.jpg",
      "prediction": "DEEPFAKE",
      "confidence": 0.9632,
      "model": "resnet50",
      "model_version": "1.0.0",
      "processing_time_ms": 782.4,
      "original_image_url": "/storage/uploads/a1b2c3d4.jpg",
      "processed_image_url": "/storage/processed/e5f6g7h8.jpg",
      "heatmap_url": "/storage/heatmaps/i9j0k1l2.jpg",
      "overlay_url": "/storage/overlays/m3n4o5p6.jpg",
      "created_at": "2026-10-10 16:40:12"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 10,
  "total_pages": 1
}
```

---

## 5. Single Prediction & Deletion

### `GET /api/v1/predictions/{id}`
Retrieve complete detail of a single prediction report.

### `DELETE /api/v1/predictions/{id}`
Deletes database record and associated image files from disk.

**Response (200 OK):**
```json
{
  "message": "Prediction #142 deleted successfully."
}
```

---

## 6. Dashboard Statistics

### `GET /api/v1/stats`
Returns aggregated analytics metrics calculated from database records.

**Response (200 OK):**
```json
{
  "total_analyses": 248,
  "real_count": 136,
  "deepfake_count": 112,
  "avg_confidence": 0.9421,
  "recent_predictions": [ ... ]
}
```
