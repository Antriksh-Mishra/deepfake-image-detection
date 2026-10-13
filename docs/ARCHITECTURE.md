# 🏛️ System Architecture Specification

## Overview

The Deepfake Image Detection Platform is organized as a production monorepo containing three primary subsystems:
1. **Machine Learning Core (`ml/`)**: Standalone data pipeline, PyTorch model definitions, training loops, benchmark evaluation, and Grad-CAM explainability.
2. **Backend API Service (`backend/`)**: High-performance FastAPI server with SQLAlchemy ORM, static file serving, validation middleware, and cached model inference.
3. **Frontend Dashboard (`frontend/`)**: Modern React 18 application built with Vite, TypeScript, Tailwind CSS, Lucide Icons, and Recharts.

---

## Data Flow Diagram

```text
User Image Upload (JPG, PNG, WEBP)
           │
           ▼
[ Frontend: React / Vite Dropzone ]
           │
           │ (POST /api/v1/predict multipart/form-data)
           ▼
[ Backend: FastAPI Router & Error Middleware ]
           │
           ├── Validate MIME, Size, and Extension
           ▼
[ ML Preprocessing Service ]
           │
           ├── OpenCV Haar Cascade Face Detection
           ├── Margin Padding & Face Bounding Box Cropping
           └── Normalization & Tensor Conversion (3, 224, 224)
           │
           ▼
[ PyTorch Inference Engine (ResNet50 / EfficientNet) ]
           │
           ├── Model Forward Pass (Loaded once at startup)
           └── Softmax Probabilities -> "REAL" vs "DEEPFAKE"
           │
           ▼
[ Explainable AI (XAI / Grad-CAM) ]
           │
           ├── Backward Activation Hook on Target Conv Layer
           ├── Global Average Pooling of Gradients
           ├── Heatmap Normalization & JET Color Mapping
           └── Weighted Image Overlay Generation
           │
           ▼
[ SQLAlchemy Database & Disk Storage ]
           │
           ├── Persist Upload, Crop, Heatmap, Overlay to disk
           └── Store Prediction Record in SQLite / PostgreSQL
           │
           ▼
[ JSON API Response ➔ React Dashboard View ]
```

---

## Design Principles
- **Separation of Concerns**: ML models and training logic can be run and updated independently of the backend API.
- **Model Lifecycle Optimization**: Deep learning models are loaded into RAM/GPU once at application startup, avoiding expensive per-request disk reads.
- **Non-Destructive Augmentations**: Training transforms preserve forensic frequency artifacts.
- **Explainability Transparency**: Grad-CAM outputs are accompanied by clear disclaimers detailing that heatmaps visualize decision activation weights rather than proof of standalone pixel tampering.
