# System Architecture — Deepfake Forensic Sentinel

**Author:** Antriksh Mishra (Final Year B.Tech CSE Capstone)  
**Architecture Identifier:** `DualDomainSentinel-v1.0`  
**Verified Production Checkpoint:** [`checkpoints/best_model.pth`](../checkpoints/best_model.pth) (SHA-256: `e8c3f62ac628323f0a3daee52923eee611da7850bf7c05c5481dc44ff49cde00`)

---

## 1. End-to-End Pipeline Architecture

Deepfake Forensic Sentinel decouples **neural forensic prediction**, **advisory explainability**, **validation-calibrated uncertainty abstention**, and **append-only software audit logging** into distinct modular layers:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   1. MEDIA INTAKE & CRYPTOGRAPHIC HASHING                  │
│  Image (JPG/PNG/WEBP/BMP/L/RGBA/CMYK)  OR  Video (.mp4 / .avi / .mov)      │
│  -> Compute streaming raw-byte SHA-256 + upload event (`audit_events`)     │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼ (Image Mode)                                          ▼ (Video Mode, temporal_modeling=False)
┌────────────────────────────────────┐        ┌───────────────────────────────────────────────────┐
│ FaceExtractor (OpenCV Haar+Eyes    │        │ OpenCV Sampling (fps / interval / uniform)         │
│ + YCrCb Skin Filter, 256x256 Crop) │        │ -> Per-frame FaceExtractor (256x256 Crop)          │
└─────────────────┬──────────────────┘        └─────────────────────────┬─────────────────────────┘
                  │                                                     │
                  └──────────────────────────┬──────────────────────────┘
                                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│            2. DUAL-DOMAIN PREPROCESSING & BATCHED NEURAL INFERENCE         │
│  SPATIAL: ImageNet RGB Norm (B, 3, 256, 256)                               │
│  SPECTRAL: Grayscale -> Shifted 2D DFT Log-Magnitude (B, 1, 256, 256)      │
│                                                                            │
│  safe_batched_forward(device: MPS -> CUDA -> CPU fallback)                 │
│    ├── SpatialBackbone: EfficientNet-B0 trunk (1280-d) -> Proj (512-d)     │
│    ├── FrequencyStream: 3 Fixed 5x5 SRM Filters + 4-stage ConvNet -> 256-d │
│    ├── CrossAttentionFusion: 4-Head MHA (Q=Spatial, K,V=Spectral) -> 256-d │
│    └── MLP Classifier: 256 -> 128 -> 32 -> 1 raw logit z                   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│       3. VALIDATION-CALIBRATED 3-OUTCOME DECISION POLICY (ZERO OVERRIDE)   │
│  P_cal = sigmoid(z / T), T = 2.654271 (fit on data/val ONLY)               │
│  Quality Guards: min Laplacian sharpness = 15.0, min dimension = 80 px     │
│  Decision Zone (Strict Invariant: P_fake > 50% can NEVER resolve to REAL):  │
│    P_cal < u_low AND P_cal <= 0.50 -> REAL / AUTHENTIC                     │
│    P_cal > u_high -> FAKE / MANIPULATED                                    │
│    u_low <= P_cal <= u_high, OR (P_cal > 0.50 & P_cal <= u_high),          │
│      OR quality failure -> UNCERTAIN                                       │
│  [Video Aggregation]: 10% Trimmed Mean + Sustained Quorum >= max(2, 30% N) │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ 4A. DECOUPLED EXPLAINABILITY (XAI)   │    │ 4B. SOFTWARE AUDIT & PDF CERTIFICATES│
│ - Per-face Grad-CAM (conv_head)      │    │ - SQLite `forensic_examinations`     │
│ - 2D FFT Inferno & Anomalous Bands   │    │ - Immutable `audit_events` Triggers  │
│ - Azimuthal Radial Profile Comparison│    │ - Multi-Page ReportLab PDF Engine    │
│ - JPEG ELA (Q=90 & Q=95/90/75 sweep) │    └──────────────────────────────────────┘
└──────────────────────────────────────┘
```

---

## 2. Neural Architecture (`DualDomainSentinelNet`)

Defined in [`src/models/dual_domain_net.py`](../src/models/dual_domain_net.py), `DualDomainSentinelNet` supports four structural operating modes via `fusion_type` (`cross_attention`, `concat`, `spatial_only`, `frequency_only`):

### 2.1 Verified Parameter Breakdown (`cross_attention` Production Model)

| Submodule | File & Class | Layer Specification | Total Params | Trainable (Frozen Trunk) |
| :--- | :--- | :--- | :---: | :---: |
| **Spatial Feature Extractor** | `SpatialBackbone.features` ([`spatial_backbone.py`](../src/models/spatial_backbone.py)) | `torchvision` EfficientNet-B0 convolutional stages + `AdaptiveAvgPool2d((1, 1))` (`1280`-d pooled) | 4,007,548 | 0 (frozen in `train.py`) |
| **Spatial Projection** | `SpatialBackbone.proj` | `Linear(1280, 512) -> LayerNorm(512) -> ReLU` | 656,896 | 656,896 |
| **Frequency & SRM Stream** | `FrequencyStream` ([`frequency_stream.py`](../src/models/frequency_stream.py)) | `SRMConv` (3 fixed `5x5` kernels as non-trainable buffers) + 4-stage stride-2 `Conv2d-BN-LeakyReLU(0.2)` (`4->32->64->128->256`) + `AdaptiveAvgPool2d` + `Linear(256, 256)->LayerNorm->ReLU` | 455,968 | 455,968 |
| **Cross-Attention Fusion** | `CrossAttentionFusion` | `Linear(512->256)`, `Linear(256->256)`, 4-head `nn.MultiheadAttention(256, 4)`, `LayerNorm` + residual `Linear(256->512)->GELU->Dropout(0.2)->Linear(512->256)` | 724,224 | 724,224 |
| **Classification Head** | `DualDomainSentinelNet.classifier` | `Linear(256->128)->LayerNorm->GELU->Dropout(0.3)->Linear(128->32)->LayerNorm->GELU->Dropout(0.15)->Linear(32->1)` | 37,377 | 37,377 |
| **Total Production Model** | `DualDomainSentinelNet` | Complete end-to-end dual-domain network | **5,882,013** | **1,874,465** |

*Note:* Non-parameter persistent buffers in `state_dict` (BatchNorm running stats + 3 fixed `5x5` SRM filter tensors) account for `457,825` elements (`6,339,838` total elements in `checkpoints/best_model.pth`).

### 2.2 Face Extraction & Validation (`FaceExtractor`)

Defined in [`src/core/face_extractor.py`](../src/core/face_extractor.py):
- Primary frontal detector: OpenCV `haarcascade_frontalface_default.xml` (`scaleFactor=1.1`, `minNeighbors=6`, adaptive `minSize >= 45px`).
- False-positive suppression (`_is_valid_face_candidate`, Phase 7 B4 fix): rejects boxes `< 50px`, non-facial aspect ratios outside `[0.70, 1.40]`, small clutter `< 18%` of dominant face area when a large face exists, and enforces an unconditional YCrCb skin-chroma check (`Cr in [130, 180], Cb in [70, 135] >= 14%`) on color images alongside `haarcascade_eye.xml` (`minNeighbors=3`) so clothing text/patterns are rejected.
- Multi-face support: `extract_all_faces()` sorts detected faces largest-first, crops square bounding boxes with `25%` contextual margin (`LANCZOS` resize to `256x256`), and falls back cleanly to a square center crop when zero faces are found.

---

## 3. Frame-Level Video Architecture (`VideoForensicAnalyzer`)

Defined in [`src/video/video_analyzer.py`](../src/video/video_analyzer.py):
- Explicitly declares **`temporal_modeling = False`** in both code and every result/certificate payload.
- Supports `.mp4`, `.avi`, and `.mov` up to `250 MB` (`validate_video_file`).
- Supports 3 deterministic frame sampling modes (`fps`, `interval`, `uniform`) capped by configurable `max_frames`.
- Extracts faces per sampled frame, batches valid face crops into `safe_batched_forward()`, applies validation temperature `T = 2.654271` and policy thresholds `[0.7070, 0.7188]`, computes `extract_suspicious_intervals()`, and applies `aggregate_video_scores()` with single-frame false-positive immunity.

---

## 4. SQLite Persistence & Immutable Audit Trail Architecture

Defined in [`src/core/database.py`](../src/core/database.py), persisted at `data/forensic_history.db` (or `DFAKE_DB_PATH` override for test isolation):

### 4.1 `forensic_examinations` Table (27 Columns, Schema v2)

| Column Group | Columns |
| :--- | :--- |
| **Core Identity & Hashes** | `case_id` (PK, `DFS-...`), `evidence_id` (`EVD-...`), `timestamp`, `image_name`, `media_type`, `image_sha256` (aligned crop PNG), `raw_file_sha256` (raw uploaded file bytes), `file_size_bytes`, `media_dimensions` |
| **Calibrated Predictions** | `verdict_tier`, `prediction_label`, `prediction_outcome`, `fake_probability`, `confidence_percent`, `decision_threshold`, `calibrated_temperature`, `processing_status` |
| **Decoupled Auxiliary & Runtime** | `neural_score`, `spectral_score`, `ela_score`, `latency_ms`, `dominant_region`, `has_deconv_spikes`, `device`, `model_version` |
| **Lifecycle & Metadata** | `is_archived` (0/1 two-step soft archive), `metadata_json` (video summary / intervals / multi-face details) |

### 4.2 Append-Only `audit_events` Table (8 Columns + Immutability Triggers)

- Tracks lifecycle events: `upload`, `processing`, `prediction`, `report_generation`, `archive_action`.
- Columns: `event_id` (PK), `evidence_id`, `case_id`, `event_type`, `event_timestamp_utc`, `sha256_at_event`, `actor`, `details_json`.
- **Database Triggers:** `trg_audit_events_no_update` and `trg_audit_events_no_delete` invoke `RAISE(ABORT, ...)` on any SQL `UPDATE` or `DELETE` attempt against `audit_events`.
- Two-step soft archiving sets `is_archived = 1` on active records while keeping `audit_events` permanently intact.

---

## 5. Unified 6-Section Streamlit Studio (`app/app.py`)

| Section | Handler / Module | Core Functionality |
| :--- | :--- | :--- |
| **1. Overview** | `render_overview_section()` ([`app/app.py`](../app/app.py)) | Executive workflow cards, live calibration readiness (`T`, `tau`, uncertainty band), dataset audit snapshot (`2,400` images, `0` leaked groups), SQLite ledger counters, quick-start navigation. |
| **2. Image Analysis** | `render_image_analysis_section()` ([`app/app.py`](../app/app.py)) | File upload, webcam capture, 8-image Benchmark Gallery, live sidebar threshold override, 3-outcome verdict banner, multi-face grid, decoupled signal cards, Unified 8-Item XAI Panel, and PDF generation. |
| **3. Video Analysis** | `render_video_analysis_view()` ([`app/video_dashboard.py`](../app/video_dashboard.py)) | MP4/AVI/MOV uploader, sampling controls (`fps`/`interval`/`uniform`, `max_frames`), Red/Green/Yellow/Gray interactive timeline, probability histogram, suspicious intervals table, clickable frame-to-XAI deep inspection, JSON/MD/PDF exports. |
| **4. Model Performance** | `render_performance_dashboard()` ([`app/performance_dashboard.py`](../app/performance_dashboard.py)) | Real test vs validation split metrics, confusion matrices & ROC curves, calibration ECE/Brier/reliability & PR threshold plots, 5-experiment ablation charts, dataset audit summary, and external benchmark status (`0/3` local). |
| **5. Forensic Evidence** | `render_forensic_evidence_section()` ([`app/app.py`](../app/app.py), [`app/xai_panel.py`](../app/xai_panel.py)) | Tab 1: Active Image or Video-Frame deep XAI (Grad-CAM regional scores, FFT real-vs-suspicious profile comparison, ELA Q=95/90/75 sweep + caveats). Tab 2: SQLite ledger, `audit_events` lifecycle viewer, raw file SHA-256 tamper verifier, CSV export, two-step archive. |
| **6. Model Information** | `render_model_information_section()` ([`app/app.py`](../app/app.py)) | Dynamic live module parameter counts (`5,882,013`), checkpoint SHA-256 provenance, live CPU RSS & MPS/CUDA memory telemetry (`get_runtime_memory_telemetry`), runtime device info, and scientific limitations disclosure. |

---

## 6. Batched Inference & Hardware Execution Architecture

Implemented in [`src/core/utils.py`](../src/core/utils.py):
- `get_device("auto")` prioritizes Apple Silicon `mps` -> NVIDIA `cuda` -> `cpu`.
- `safe_batched_forward()` runs stacked tensor mini-batches `(N, 3, 256, 256)` and `(N, 1, 256, 256)` on the active accelerator; if an accelerator `RuntimeError` (e.g., OOM or unsupported op) occurs, it clears cache via `clear_accelerator_cache()`, moves the model safely to CPU, completes inference, and records `cpu_fallback_triggered=True`.
- `get_evaluation_artifacts_signature()` in [`app/performance_dashboard.py`](../app/performance_dashboard.py) tracks `(path, st_mtime_ns, st_size)` across 14 artifact files for instant cache invalidation on updates (**B10 fix**).
- Non-interactive Matplotlib `Agg` backend is forced across all visualization modules so multi-threaded Streamlit and pytest runs never trigger macOS GUI thread exceptions.

---

## 7. Complete Repository Module Map

```text
dfake/
├── app/
│   ├── app.py                     # Unified 6-section Streamlit Studio entry point
│   ├── performance_dashboard.py   # Empirical metrics, calibration, ablation & mtime cache signature
│   ├── report_generator.py        # Multi-page ReportLab Image & Video PDF certificate engine
│   ├── video_dashboard.py         # Interactive video forensic timeline & frame inspector UI
│   └── xai_panel.py               # Unified 8-item XAI panel, Grad-CAM, FFT comparison & ELA tabs
├── config/
│   └── default_config.yaml        # Hyperparameters, forensic quality & external dataset roots
├── src/
│   ├── pipeline.py                # Calibrated multi-face batched inference + decoupled XAI orchestration
│   ├── train.py                   # Dual-domain training loop (frozen trunk + LabelSmoothingBCELoss)
│   ├── core/
│   │   ├── database.py            # SQLite v2 ledger + immutable `audit_events` triggers + tamper check
│   │   ├── face_extractor.py      # OpenCV Haar + eye + YCrCb face detection & 256x256 alignment
│   │   └── utils.py               # Device fallback, accelerator cache, memory telemetry & transforms
│   ├── data/
│   │   ├── dataset.py             # DualDomainDataset loader with corruption tracking & strict split checks
│   │   ├── dataset_audit.py       # SHA-256 + 64-bit 2D DCT pHash + StratifiedGroupKFold auditor
│   │   ├── external_datasets.py   # FaceForensics++, Celeb-DF & DFDC standardized sample loaders
│   │   └── generate_realistic_dataset.py # Source-disjoint synthetic portrait benchmark generator
│   ├── evaluation/
│   │   ├── ablation_study.py      # 5-experiment group-disjoint ablation benchmark runner
│   │   ├── calibration.py         # Validation-only Temperature Scaling, Youden's J, ECE/Brier & 3-outcome policy
│   │   ├── evaluate_checkpoint.py # Batched checkpoint evaluation & ROC/CM plotting
│   │   ├── external_benchmark.py  # Zero-tuning external generalization evaluator
│   │   ├── metrics.py             # Binary accuracy, precision/recall/F1, ROC-AUC, PR-AUC & EER
│   │   └── robustness_bench.py    # Batched JPEG, Gaussian blur & Gaussian noise perturbation harness
│   ├── explainability/
│   │   ├── ela.py                 # JPEG Q=90 & Q=95/90/75 ELA maps, block CV & explicit caveats
│   │   ├── gradcam.py             # Autograd spatial Grad-CAM hooks (+logit/-logit) & regional pooling
│   │   └── spectral_analysis.py   # Shifted 2D DFT, azimuthal average, anomaly rings & ref comparison
│   ├── models/
│   │   ├── dual_domain_net.py     # DualDomainSentinelNet + CrossAttentionFusion (4 fusion modes)
│   │   ├── frequency_stream.py    # 3 fixed 5x5 SRM kernels + 4-stage ConvNet spectral encoder
│   │   ├── loss.py                # LabelSmoothingBCELoss & BinaryFocalLoss
│   │   └── spatial_backbone.py    # EfficientNet-B0 / ResNet-34 spatial trunk & projection
│   └── video/
│       └── video_analyzer.py      # Frame-level MP4/AVI/MOV decoder, batched quorum & interval extraction
├── results/                       # Verified JSON audits, calibration diagrams & external check summaries
├── checkpoints/                   # Preserved production checkpoint + ablation checkpoints + plots
├── tests/                         # 14 pytest modules (73 verified unit, integration & AppTest E2E tests)
└── docs/                          # Complete engineering, mathematical, evaluation & capstone documentation
```
