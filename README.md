---
title: Deepfake Image Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ Deepfake Image Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2%2B-blue.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2%2B-blue.svg)](https://www.typescriptlang.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-grade, full-stack AI platform designed to detect manipulated and synthetic deepfake images with high accuracy. The platform couples deep learning feature extraction (PyTorch ResNet50 / EfficientNet) with Explainable AI (**Grad-CAM**), a resilient **FastAPI** backend, and an intuitive **React/TypeScript SaaS dashboard**.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [Technology Stack](#-technology-stack)
4. [Monorepo Directory Layout](#-monorepo-directory-layout)
5. [Machine Learning Pipeline](#-machine-learning-pipeline)
6. [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
7. [REST API Specification](#-rest-api-specification)
8. [Getting Started Locally](#-getting-started-locally)
9. [Docker Deployment](#-docker-deployment)
10. [Automated Testing Suite](#-automated-testing-suite)
11. [Ethical Considerations & Limitations](#-ethical-considerations--limitations)
12. [Future Improvements](#-future-improvements)

---

## 🌟 Key Features

- **Automated Face Localization**: Isolates and crops primary human faces with margin padding using OpenCV Haar Cascades before classification.
- **Deep Learning Architectures**: Dynamic model factory supporting Custom CNN baseline, ResNet50, and EfficientNet-B0 with custom binary classification heads.
- **Explainable AI (Grad-CAM)**: Visualizes convolutional feature activation weights, rendering raw heatmaps and blended overlays highlighting influential image regions.
- **High-Performance Inference**: PyTorch models are loaded once into RAM/GPU at application startup to eliminate per-request overhead.
- **Interactive SaaS Dashboard**:
  - **Detection Studio**: Drag-and-drop file upload with live preview, size caps, and instant analysis.
  - **Inspection Studio**: Side-by-side tabbed inspector (`Overlay`, `Heatmap`, `Face Crop`, `Original`).
  - **History Log**: Searchable, filterable (Real vs Deepfake), and paginated database log with detail modal drawer and record deletion.
  - **Analytics**: Recharts visualizations for detection volume ratios and confidence score distributions.
- **Production Architecture**: Strict separation of concerns, SQLite for development, PostgreSQL-ready SQLAlchemy schema, structured JSON error codes, and Docker Compose orchestration.

---

## 🏛️ System Architecture

```text
[ User Browser ] 
       │ (React 18 + TypeScript + Vite + Tailwind CSS)
       ▼
[ FastAPI Backend API ] ─── (CORS & Error Middleware)
       │
       ├── Preprocessing Service (OpenCV Face Detection & Crop, Normalization)
       ├── PyTorch Inference Engine (Model loaded once in RAM/GPU)
       ├── Grad-CAM Visualizer (JET Colormap Overlay Generator)
       ├── Disk Storage (/storage/uploads, /storage/processed, /storage/heatmaps)
       └── SQLAlchemy Database (SQLite / PostgreSQL)
```

---

## 🛠️ Technology Stack

| Domain | Technologies |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts, Axios, React Router |
| **Backend** | Python 3.10+, FastAPI, Pydantic, SQLAlchemy, Uvicorn |
| **Machine Learning** | PyTorch, torchvision, OpenCV, NumPy, Pillow, scikit-learn, Captum / Grad-CAM |
| **Database** | SQLite (local development), PostgreSQL-compatible SQLAlchemy models |
| **DevOps** | Docker, Docker Compose, `.env` Configuration |

---

## 📁 Monorepo Directory Layout

```text
deepfake-image-detection/
├── frontend/             # Vite + React + TypeScript + Tailwind CSS application
│   ├── src/
│   │   ├── components/   # Navbar, ResultView, ToastContainer
│   │   ├── pages/        # DashboardPage, DetectionPage, HistoryPage, AnalyticsPage
│   │   ├── services/     # api.ts (Central Axios API client)
│   │   └── types/        # api.ts (TypeScript interface contracts)
│   ├── package.json
│   └── vite.config.ts
├── backend/              # FastAPI application server
│   ├── app/
│   │   ├── config/       # Pydantic Settings configuration
│   │   ├── models/       # SQLAlchemy database models & session manager
│   │   ├── routes/       # REST API endpoint routers
│   │   ├── schemas/      # Pydantic request/response DTOs
│   │   ├── services/     # InferenceService & StorageService
│   │   └── middleware/   # Custom exception handling middleware
│   ├── requirements.txt
│   └── run.py
├── ml/                   # Machine Learning pipeline
│   ├── configs/          # default.yaml hyperparameter configs
│   ├── models/           # Checkpoints (best_model.pth, model_config.json)
│   └── src/
│       ├── preprocessing/# FaceDetector, ImagePreprocessor, exceptions
│       ├── datasets/     # DeepfakeDataset & DataLoader factory
│       ├── models/       # CustomCNN, ResNet50Detector, EfficientNetDetector
│       ├── training/     # DeepfakeTrainer (Early stopping, LR scheduling)
│       ├── evaluation/   # ModelEvaluator & DatasetEDA
│       └── explainability/# GradCAM & ExplainabilityVisualizer
├── data/                 # Dataset documentation & guidelines
├── tests/                # Test suites & master test runner
├── docs/                 # API & Architecture documentation
├── docker/               # Dockerfiles for frontend and backend
├── docker-compose.yml    # Multi-container orchestration
└── .env.example          # Environment variable template
```

---

## 🔬 Machine Learning Pipeline

### 1. Dataset Organization
Organize images into the standard directory hierarchy:
```text
dataset/
├── train/        (real/ and fake/)
├── validation/   (real/ and fake/)
└── test/         (real/ and fake/)
```

### 2. Preprocessing & Augmentation
- Input validation: verifies MIME type and enforces the 10 MB limit.
- Face Detection: OpenCV Haar Cascade identifies human faces and crops with margin padding.
- Normalization: standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- Augmentation: Conservative training transforms (subtle rotation, horizontal flip, slight color jitter) prevent degradation of forensic artifacts.

### 3. Model Training
Train classifiers with checkpointing and early stopping:
```bash
python3 ml/src/training/trainer.py
```
Checkpoints are saved to `ml/models/best_model.pth` alongside `model_config.json` and `class_mapping.json`.

---

## 👁️ Explainable AI (Grad-CAM)

For every processed image, Grad-CAM computes activation weights against target convolutional layers (`layer4` for ResNet50, `features` for EfficientNet). The activation maps are converted to OpenCV JET colormaps and blended onto the face crop.

> **Explainability Notice**: Grad-CAM visualizes region influence on model decision scores. Highlighted areas indicate features that contributed most to the prediction score and do not constitute definitive proof that a specific region was independently manipulated.

---

## 📡 REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health status |
| `GET` | `/api/v1/model` | Deployed model parameters and configuration |
| `POST` | `/api/v1/predict` | Upload image, run inference, generate Grad-CAM, persist record |
| `GET` | `/api/v1/predictions` | Paginated, searchable, filterable history log |
| `GET` | `/api/v1/predictions/{id}` | Retrieve single prediction report |
| `DELETE` | `/api/v1/predictions/{id}` | Delete prediction record and associated disk files |
| `GET` | `/api/v1/stats` | Aggregate dashboard statistics from database |

Full API reference is available in [docs/API_DOCUMENTATION.md](file:///Users/antrikshmishra/Desktop/deepfake/docs/API_DOCUMENTATION.md).

---

## 🚀 Getting Started Locally

### 1. Clone & Configure
```bash
cp .env.example .env
```

### 2. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```
Backend will start on `http://localhost:8000`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend dashboard will be accessible at `http://localhost:5173`.

---

## 🐳 Docker Deployment

Run the complete multi-container stack with a single command:
```bash
docker compose up --build
```
- **Frontend Dashboard**: `http://localhost:5173`
- **FastAPI Documentation**: `http://localhost:8000/docs`

---

## 🧪 Automated Testing Suite

Execute all platform unit and integration tests:
```bash
python3 tests/run_all_tests.py
```

Test coverage includes:
- `tests/test_preprocessing.py`: Image validation, face cropper, custom exceptions.
- `tests/test_models.py`: PyTorch forward passes, model factory, evaluation metric calculations.
- `tests/test_explainability.py`: Grad-CAM activation mapping, heatmap generation, overlay blending.
- `tests/test_api.py`: FastAPI endpoints, database operations, error codes.
- `tests/test_e2e_pipeline.py`: Complete End-to-End flow verification.

---

## ⚠️ Ethical Considerations & Limitations

1. **Not Absolute Proof**: A high classification confidence score reflects pattern similarities with training datasets, not mathematical certainty.
2. **Compression Artifacts**: Heavy social media compression (JPEG re-compression) can introduce artifacts that occasionally trigger false positives.
3. **Face Cropping Requirement**: The platform is optimized for human facial portraits. Non-facial manipulations will be rejected by the face detection filter.
4. **Generalization**: Detection performance may vary across generative architectures (GAN vs Diffusion) not present in the training distribution.

---

## 🔮 Future Improvements

- **Video Deepfake Pipeline**: Frame-by-frame temporal consistency analysis using 3D CNNs or LSTM/GRU networks.
- **Frequency-Domain Features**: Discrete Cosine Transform (DCT) and Fast Fourier Transform (FFT) analysis to uncover invisible generative spectrum artifacts.
- **Multimodal Audio-Visual Sync**: Detecting voice clone desynchronization in talking-head videos.
- **Diffusion-Specific Detectors**: Specialization against newer diffusion models (Stable Diffusion, Midjourney, Flux).
