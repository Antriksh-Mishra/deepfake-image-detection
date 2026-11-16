# Deepfake Forensic Sentinel

> **Dual-Domain Spatial & Frequency Deepfake Detection Suite with Calibrated Uncertainty, Frame-Level Video Analysis, Explainable AI, and Software Forensic Audit Trails**  
> **Lead Developer & Researcher:** Antriksh Mishra (Final Year B.Tech Capstone — Computer Science & Engineering)

[![Python 3.12](https://img.shields.io/badge/Python-3.10%2B%20(Verified%203.12.6)-blue.svg)](https://www.python.org/)
[![PyTorch 2.15](https://img.shields.io/badge/PyTorch-2.x%20MPS%20%7C%20CUDA%20%7C%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit 1.67](https://img.shields.io/badge/Streamlit-6--Section%20Studio-ff4b4b.svg)](https://streamlit.io/)
[![Tests 79/79](https://img.shields.io/badge/Pytest-79%20Passing-16a34a.svg)](#testing--quality-assurance)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

---

## Executive Summary

**Deepfake Forensic Sentinel** is a calibrated dual-domain deepfake inspection platform combining spatial facial representations (`EfficientNet-B0`) with spectral/steganalysis cues (`2D FFT` log-magnitude + 3 fixed `5x5` Spatial Rich Model high-pass filters) via 4-head Multi-Head Cross-Attention (`DualDomainSentinelNet`, `5,882,013` parameters).

Across a 12-phase engineering and scientific hardening lifecycle, the project eliminates artificial dataset leakage, separates auxiliary forensic cues from classifier logits, applies validation-only temperature scaling and 3-outcome abstention (`AUTHENTIC`, `MANIPULATED`, `UNCERTAIN / NEEDS MANUAL REVIEW`), extends inference to frame-level MP4/AVI/MOV video timelines, upgrades multi-face explainability, enforces append-only SQLite audit event triggers, optimizes batched MPS/CUDA inference, and ships with full unit, integration, and Streamlit UI verification (`73/73` pytest checks).

---

## Core Capabilities

| Capability | Implementation Summary | Primary Module(s) |
| :--- | :--- | :--- |
| **Dual-Domain Neural Detector** | Frozen/fine-tunable `EfficientNet-B0` (`512`-d spatial proj) + 3 fixed `5x5` SRM residuals over 2D FFT (`256`-d spectral ConvNet) + 4-head `CrossAttentionFusion` (`256`-d) -> MLP classifier (`5,882,013` total params, `1,874,465` trainable under frozen trunk). | [`src/models/dual_domain_net.py`](src/models/dual_domain_net.py), [`src/models/spatial_backbone.py`](src/models/spatial_backbone.py), [`src/models/frequency_stream.py`](src/models/frequency_stream.py) |
| **Validation-Only Calibration & Strict 3-Outcome Policy** | Fits scalar temperature `T = 2.654271` and Youden's J threshold `tau = 0.71` strictly on `data/val` (`zero_test_set_tuning = true`), enforcing strict mathematical invariant ($P(\text{Fake}) > 50\%$ can never resolve to `REAL`) with `REAL`, `FAKE`, and `UNCERTAIN` verdict states. | [`src/evaluation/calibration.py`](src/evaluation/calibration.py), [`src/pipeline.py`](src/pipeline.py) |
| **Intelligent Media Localization & Context Awareness** | Automatic Haar frontal face detection with YCrCb skin chrominance validation, paired with salient contour boundary detection for non-face objects (sneakers, products, animals, vehicles, digital art), dynamically disabling facial terminology on non-face inputs. | [`src/core/face_extractor.py`](src/core/face_extractor.py) |
| **Spatial Rich Model (SRM) Noise Analysis** | Standalone 2D directional high-pass filtering ($F_1, F_2, F_3$) with reflection padding, residual energy, residual kurtosis (Gaussian baseline $\approx 3.0$), and $16 \times 16$ block CV spatial noise inconsistency thermal mapping. | [`src/explainability/srm_analysis.py`](src/explainability/srm_analysis.py) |
| **Adaptive 2D FFT & Dynamic Optical Falloff** | Centered 2D DFT log-magnitude frequency transform dynamically fitting power-law $1/f^\alpha$ optical decay to input low-frequencies ($r \in [2, 35]$) with discrete frequency radius reporting ($r=X\text{ px}$). | [`src/explainability/spectral_analysis.py`](src/explainability/spectral_analysis.py) |
| **Container-Aware Error Level Analysis** | Automatic detection of source container formats (PNG, WEBP, JPEG), qualifying recompression differences for lossless containers and isolating foreground vs. background quadrant compression error. | [`src/explainability/ela.py`](src/explainability/ela.py) |
| **Leakage Audit & Group-Disjoint Splits** | SHA-256 exact hash checks + pure SciPy 64-bit 2D DCT perceptual hashing (pHash) + `StratifiedGroupKFold` manifest generation across `2,400` images and `15` source groups (`0` leaked groups). | [`src/data/dataset_audit.py`](src/data/dataset_audit.py) |
| **Controlled 5-Experiment Ablation** | Evaluates `spatial_only`, `frequency_only`, `concat`, `cross_attention`, and `LabelSmoothingBCELoss(0.06)` over identical group-disjoint splits (`1,426 / 488 / 486`). | [`src/evaluation/ablation_study.py`](src/evaluation/ablation_study.py) |
| **Multi-Category Out-of-Sample Benchmark** | Stratified evaluation across 11 visual domains: real photos, AI images, diffusion faces, StyleGAN checkerboard, face-swaps, products, animals, scenes, compressed, resized, and digital screenshots. | [`src/evaluation/category_eval.py`](src/evaluation/category_eval.py) |
| **Frame-Level Video Forensics** | OpenCV MP4/AVI/MOV decoder with FPS/interval/uniform sampling, batched face inference, 10% trimmed mean + sustained quorum aggregation (`>= max(2, ceil(0.30 * N))`), suspicious timestamp intervals, and interactive Red/Green/Yellow/Gray timeline (`temporal_modeling=False`). | [`src/video/video_analyzer.py`](src/video/video_analyzer.py), [`app/video_dashboard.py`](app/video_dashboard.py) |
| **Unified 8-Item Forensic Evidence Panel** | Multi-face or structural object Grad-CAM overlays, Spatial SRM noise residuals, 2D FFT Inferno log-magnitude, azimuthal radial power comparison, and multi-quality JPEG/PNG ELA (`Q=95/90/75`). | [`src/explainability/gradcam.py`](src/explainability/gradcam.py), [`src/explainability/spectral_analysis.py`](src/explainability/spectral_analysis.py), [`src/explainability/srm_analysis.py`](src/explainability/srm_analysis.py), [`src/explainability/ela.py`](src/explainability/ela.py), [`app/xai_panel.py`](app/xai_panel.py) |
| **Software Forensic Audit Ledger & PDF Certificates** | SQLite schema v2 (`27` examination columns + append-only `audit_events` guarded by `UPDATE`/`DELETE` SQLite `BEFORE` triggers), raw uploaded file SHA-256 verification vs aligned-crop hashes, and multi-page ReportLab image & video PDF certificates. | [`src/core/database.py`](src/core/database.py), [`app/report_generator.py`](app/report_generator.py) |
| **6-Section Streamlit Studio & Zero-Gap Telemetry** | Unified workspace (`Overview`, `Image Analysis`, `Video Analysis`, `Model Performance`, `Forensic Evidence`, `Model Information`) with zero-gap latency accounting, live RSS/accelerator RAM telemetry, and mtime+size cache invalidation. | [`app/app.py`](app/app.py), [`app/performance_dashboard.py`](app/performance_dashboard.py), [`src/core/utils.py`](src/core/utils.py) |

---

## System Architecture

```
Input Image / Video (.mp4, .avi, .mov)
   │
   ├─► Raw Cryptographic SHA-256 Digest + Upload Event (`audit_events`)
   ├─► [If Video] OpenCV Sampling (fps / interval / uniform) -> Frame Batching (`temporal_modeling=False`)
   ▼
FaceExtractor (OpenCV Haar + Eye Cascade + YCrCb Skin Validation -> 256x256 Crop / Center Fallback)
   │
   ├──────────────────────────────────┬──────────────────────────────────┐
   ▼                                  ▼                                  ▼
[Spatial Stream]               [Frequency Stream]               [Decoupled XAI Aids]
EfficientNet-B0 Trunk          2D DFT Log-Magnitude (1ch)       - Spatial Grad-CAM
+ Linear+LayerNorm (512-d)     + 3 Fixed 5x5 SRM Kernels (3ch)  - Azimuthal Radial Profile
   │                           + 4-Stage ConvNet (256-d)        - JPEG Q=90/95/75 ELA
   └──────────────┬───────────────────┘                                  │
                  ▼                                                      │
     [CrossAttentionFusion] (4-Head MHA: Q=Spatial, K,V=Spectral)        │
                  │ (256-d)                                              │
                  ▼                                                      │
     [MLP Classifier Head] (256 -> 128 -> 32 -> 1 Raw Logit z)           │
                  │                                                      │
                  ▼                                                      │
     Temperature Scaling: P_cal = sigmoid(z / 2.654271)                  │
     3-Outcome Policy: AUTHENTIC (<0.707) | UNCERTAIN [0.707, 0.7188] | MANIPULATED (>0.7188)
                  │                                                      │
                  └───────────────────────┬──────────────────────────────┘
                                          ▼
        SQLite `forensic_examinations` + Immutable `audit_events` + Multi-Page PDF Certificate
```

---

## Verified Empirical Results

> **Scientific Honesty Note:** All metrics below are taken directly from verified repository artifacts generated on our source-disjoint local benchmark (`checkpoints/best_model.pth`, SHA-256 `e8c3f62ac628323f0a3daee52923eee611da7850bf7c05c5481dc44ff49cde00`) without cherry-picking or test-set tuning.

### 1. Dataset Integrity & Production Evaluation (`N=400` Val / `N=400` Test)

- **Audit Status (`results/dataset_audit.json`):** `2,400` valid RGB `256x256` images (`1,600` train / `400` val / `400` test), `0` corrupt, `0` missing, `0` exact SHA-256 duplicates, `15` source groups with **`0` cross-split leaked groups** (`1,653` within-split augmented near duplicates via pHash `<= 4`, `0` cross-split near duplicates).

| Split & Operating Point | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | EER | NLL | Brier | ECE (Pos / Conf) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (`N=400`, Uncalibrated `tau=0.50`)** | 69.00% | 0.6638 | 0.7700 | 0.7130 | 0.7694 | 0.8326 | 23.00% | 0.7728 | 0.2522 | 0.2364 / 0.2098 |
| **Validation (`N=400`, Calibrated `T=2.6543, tau=0.71`)** | 77.25% | 0.7853 | 0.7500 | 0.7673 | 0.7694 | 0.8326 | 23.00% | 0.6097 | 0.2098 | 0.1142 / 0.0484 |
| **Held-Out Test (`N=400`, Uncalibrated `tau=0.50`)** | 75.50% | 0.9811 | 0.5200 | 0.6797 | 0.7396 | 0.8295 | 32.00% | 0.6247 | 0.2011 | 0.1867 / 0.1488 |
| **Held-Out Test (`N=400`, Calibrated `T=2.6543, tau=0.71`)** | 69.25% | 1.0000 | 0.3850 | 0.5560 | 0.7396 | 0.8295 | 32.00% | 0.5571 | 0.1850 | 0.0983 / 0.0655 |

- **3-Outcome Validation Selective Policy (`[0.707, 0.7188]`):** Achieves `82.69%` selective accuracy at `83.75%` coverage on validation (`16.25%` routed to `UNCERTAIN`). On out-of-sample held-out test, `7.50%` (`30/400`) route to `UNCERTAIN`, while conservative upper boundary `>0.7188` preserves `0` false positives (`51/51` precision on manipulated calls; higher FNR due to domain shift).

### 2. Controlled 5-Experiment Ablation Study (`results/ablation_results.json`)

Evaluated on identical `StratifiedGroupKFold` splits (`1,426` train / `488` val / `486` test, `8` epochs on Apple Silicon MPS):

| Experiment | Fusion Mode | Loss Function | Parameters | Test Acc | Test F1 | Test AUC | Test EER | Latency (`bs=1`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1** | `spatial_only` | `BCEWithLogitsLoss` | 4,734,589 | 59.05% | 0.700 | 0.8251 | 25.51% | 9.27 ms |
| **Exp 2** | `frequency_only` | `BCEWithLogitsLoss` | 493,345 | **80.04%** | 0.777 | **0.9092** | 20.99% | 12.19 ms |
| **Exp 3** | `concat` | `BCEWithLogitsLoss` | 5,223,325 | 78.60% | 0.795 | 0.8840 | 21.81% | 38.33 ms |
| **Exp 4** | `cross_attention` | `BCEWithLogitsLoss` | 5,882,013 | 76.54% | 0.788 | 0.8901 | 23.46% | 17.38 ms |
| **Exp 5** | `cross_attention` | `LabelSmoothingBCELoss(0.06)` | 5,882,013 | 79.84% | **0.822** | 0.8752 | **19.55%** | 10.11 ms |

### 3. External Benchmarks & Batched Runtime Performance

- **External Benchmarks (`results/external_benchmark/external_benchmark_summary.json`):** `FaceForensics++`, `Celeb-DF`, and `DFDC` all report `available: false` (`0/3` locally installed); loaders and evaluators are verified via unit/integration tests without fabricating external metrics.
- **Batched Inference (`N=64` neural-only benchmark):** Warmed Apple Silicon MPS `batch_size=32` runs at **`219.9 img/s`** (`291.1 ms` total), `1.60x` faster than MPS `batch_size=1` (`137.7 img/s`) and `26.4x` faster than CPU `batch_size=32` (`8.3 img/s`).

---

## Quickstart Guide

### 1. Environment Setup

```bash
cd /Users/antrikshmishra/Desktop/dfake
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the 6-Section Streamlit Studio

On sandboxed macOS or restricted home environments, pass explicit writable paths for Matplotlib, Torch Hub, and Home:

```bash
TORCH_HOME=~/.cache/torch \
HOME=/Users/antrikshmishra/Desktop/dfake \
MPLCONFIGDIR=/Users/antrikshmishra/Desktop/dfake/.matplotlib \
.venv/bin/streamlit run app/app.py --server.port 8501 --server.headless true
```

Open [http://localhost:8501](http://localhost:8501) to access **Overview**, **Image Analysis**, **Video Analysis**, **Model Performance**, **Forensic Evidence**, and **Model Information**.

### 3. CLI Evaluation, Audit & Calibration Commands

```bash
# 1. Verify dataset leakage & perceptual hash audit (read-only)
.venv/bin/python -m src.data.dataset_audit --data_dir data

# 2. Evaluate production checkpoint (batched on MPS/CUDA/CPU)
.venv/bin/python -m src.evaluation.evaluate_checkpoint --checkpoint checkpoints/best_model.pth

# 3. Fit validation-only temperature scaling & 3-outcome policy
.venv/bin/python -m src.evaluation.calibration --checkpoint checkpoints/best_model.pth

# 4. Run 5-experiment group-disjoint ablation study
.venv/bin/python -m src.evaluation.ablation_study

# 5. Evaluate any downloaded FaceForensics++ / Celeb-DF / DFDC directories
.venv/bin/python -m src.evaluation.external_benchmark --checkpoint checkpoints/best_model.pth
```

### 4. Testing & Quality Assurance

Run the full **79-test** QA suite covering neural modules, sneaker and non-face object localization, dataset auditing, external loaders, ablation, calibration, frame-level video forensics, XAI explainability, Spatial SRM residuals, SQLite immutability triggers, ReportLab PDF certificates, batched MPS/CPU fallback, and live isolated Streamlit `AppTest` 6-section execution:

```bash
TORCH_HOME=~/.cache/torch \
HOME=/Users/antrikshmishra/Desktop/dfake \
MPLCONFIGDIR=/Users/antrikshmishra/Desktop/dfake/.matplotlib \
.venv/bin/pytest tests/ -q
```

---

## Important Scientific Limitations

1. **Domain Scope:** Trained on `2,400` locally synthesized portrait variations across `15` source identities; unseen third-party manipulations (`deepfake_boundary_morph.jpg`, `deepfake_faceswap_manipulation.jpg` in demo probes) can produce false negatives until retrained/evaluated on large multi-identity benchmarks.
2. **Video Temporal Scope (`temporal_modeling=False`):** Video analysis evaluates independently sampled frames with multi-frame quorum aggregation rather than inter-frame optical flow, lip-sync audio, or 3D spatiotemporal transformers.
3. **Explainability Caveat:** Grad-CAM, 2D FFT curves, and JPEG ELA residuals are advisory diagnostic visualizations and respond to benign resizing, camera sharpening, and social-media recompression (WhatsApp/Instagram).
4. **Software Audit Log vs. Legal Chain of Custody:** The SQLite `audit_events` table provides local tamper detection and event traceability, not RFC 3161 TSA timestamping, PKI/HSM signatures, or legally attested hardware custody transfers.

---

## Documentation Suite

| Document | Scope & Contents |
| :--- | :--- |
| [`docs/PROJECT_ARCHITECTURE.md`](docs/PROJECT_ARCHITECTURE.md) | Dual-domain neural layers (`5,882,013` params), video pipeline, SQLite schema v2 + triggers, Streamlit 6-section topology, and dependency map. |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Mathematical derivations for 2D DFT radial decay, fixed 5x5 SRM kernels, 4-head cross-attention, temperature scaling, uncertainty bands, Grad-CAM, and ELA. |
| [`docs/DATASET_AND_EVALUATION.md`](docs/DATASET_AND_EVALUATION.md) | Source-disjoint dataset audit, validation/test calibration metrics, 5-experiment ablation analysis, demo probes, and external benchmark status. |
| [`docs/TESTING_REPORT.md`](docs/TESTING_REPORT.md) | Full `79/79` pytest verification, Streamlit `AppTest` UI run, MPS vs CPU batching benchmarks, and resolved bug log (`B1`–`B15`). |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Step-by-step setup, 6-section studio walkthrough, raw SHA-256 tamper verification, two-step archiving, PDF exports, and CLI reference. |
| [`docs/TECHNICAL_LIMITATIONS.md`](docs/TECHNICAL_LIMITATIONS.md) | Generalization bounds, video frame-level boundary, XAI false-positive/washout physics, audit log vs legal custody, and performance trade-offs. |
| [`docs/CAPSTONE_SUBMISSION_KIT.md`](docs/CAPSTONE_SUBMISSION_KIT.md) | Complete B.Tech Project Report chapter structure and 14-slide Capstone Defense Presentation outline. |
| [`docs/viva_questions.md`](docs/viva_questions.md) | 30 examiner-grade Viva Voce questions & answers grounded in empirical numbers and code paths. |
| [`docs/project_synopsis.md`](docs/project_synopsis.md) | Concise updated B.Tech Capstone synopsis aligned with verified deliverables. |
| [`docs/architecture_breakdown.md`](docs/architecture_breakdown.md) | Compact mathematical cheat-sheet matching production `LabelSmoothingBCELoss` and calibration formulas. |
| [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md) | Historical Phase 1 baseline audit snapshot prior to Phases 2–12 upgrades. |

---

## License

Released under the **MIT License**.
