# Project Audit & Technical Baseline — Deepfake Forensic Sentinel

**Audit Date:** 2026-11-10  
**Repository Root:** `/Users/antrikshmishra/Desktop/dfake`  
**Phase:** Phase 1 — Complete Project Inspection, Technical Audit & Baseline Backup  
**Git Status:** Not a Git repository (baseline preserved manually in `backups/phase1_baseline/`)  

> [!IMPORTANT]
> This audit was completed strictly in read-only inspection mode prior to modifying any codebase module, trained weight file, or database record. All baseline artifacts were backed up before any upcoming upgrades.

---

## 1. Existing Modules Inventory

### 1.1 Configuration & Environment
| Path | Key Symbols / Contents | Purpose |
| :--- | :--- | :--- |
| `pyproject.toml` | `[project]`, `[tool.pytest.ini_options]` | Package metadata (`deepfake-forensic-sentinel` v1.0.0) and pytest `pythonpath = ["."]` config. |
| `requirements.txt` | Core CV, Torch, Streamlit, ReportLab deps | Python 3.12 dependency specification. |
| `config/default_config.yaml` | `project`, `hardware`, `data`, `model`, `train`, `forensics` | Central hyperparameters and forensic settings. |
| `.streamlit/config.toml` | `[browser]`, `[server]`, `[theme]` | Headless Streamlit port `8501`, light blue/slate theme (`#2563eb`). |
| `.streamlit/credentials.toml` | `[general]` | Suppresses Streamlit first-run email prompt. |

### 1.2 Core Utilities, Preprocessing & Storage (`src/core/`)
| Path | Key Classes / Functions | Purpose |
| :--- | :--- | :--- |
| `src/core/utils.py` | `get_device`, `set_seed`, `load_config`, `SPATIAL_TRANSFORM`, `SPECTRAL_TRANSFORM` | Device selection (`mps` -> `cuda` -> `cpu`), deterministic seeding, ImageNet spatial (`256x256`) and 1-ch spectral (`mean=0.5, std=0.5`) normalization. |
| `src/core/face_extractor.py` | `FaceExtractor` (`detect_faces`, `extract_face`, `extract_all_faces`, `_is_valid_face_candidate`) | OpenCV Haar frontal face + eye cascade detection, YCrCb skin-chroma check, aspect-ratio filtering, `256x256` Lanczos crop with `0.25` context margin, center-crop fallback. |
| `src/core/database.py` | `compute_image_sha256`, `ForensicDatabase` (`init_db`, `log_examination`, `get_history`, `clear_history`) | SQLite persistence at `data/forensic_history.db` (table `forensic_examinations`, currently **50 existing rows**). |

### 1.3 Neural Network Architecture & Losses (`src/models/`)
| Path | Key Classes / Functions | Purpose |
| :--- | :--- | :--- |
| `src/models/spatial_backbone.py` | `SpatialBackbone` | Wraps `torchvision` `efficientnet_b0` (1280-d -> 512-d) or `resnet34` (512-d -> 512-d) with `AdaptiveAvgPool2d`, `Linear + LayerNorm + ReLU` projection, and `target_cam_layer` hook target. |
| `src/models/frequency_stream.py` | `get_srm_filters`, `SRMConv`, `FrequencyStream` | Applies 3 fixed `5x5` SRM high-pass filters to the `256x256` 2D FFT log-magnitude map (concatenated into 4 channels), followed by a 4-stage stride-2 Conv2D encoder -> `256`-d spectral embedding. |
| `src/models/dual_domain_net.py` | `CrossAttentionFusion`, `DualDomainSentinelNet` | Fuses 512-d spatial features (Queries) and 256-d spectral features (Keys/Values) via 4-head `nn.MultiheadAttention` (`embed_dim=256`, or `concat` fallback) into a 3-layer MLP binary logit classifier (`128 -> 32 -> 1`). |
| `src/models/loss.py` | `LabelSmoothingBCELoss`, `BinaryFocalLoss` | `LabelSmoothingBCELoss(smoothing=0.05)` and `BinaryFocalLoss(alpha=0.50, gamma=2.0)`. |

### 1.4 Explainability Modules (`src/explainability/`)
| Path | Key Classes / Functions | Purpose |
| :--- | :--- | :--- |
| `src/explainability/gradcam.py` | `GradCAM` (`generate_cam`, `overlay_cam`, `remove_hooks`) | Forward/backward hooks on spatial `target_cam_layer` producing `256x256` JET heatmap overlays. |
| `src/explainability/spectral_analysis.py` | `SpectralAnalyzer` (`compute_fft_magnitude`, `azimuthal_average`, `analyze_spectral_peaks`) | Centered 2D DFT log magnitude spectrum, 1D azimuthal radial average, peak detection vs $r^{-0.35}$ decay curve, high-frequency ratio, and second-difference smoothness. |
| `src/explainability/ela.py` | `ErrorLevelAnalysis` (`compute_ela`, `compute_ela_detailed`) | JPEG re-compression (`Q=90`), brightness scaling (`15x`), histogram bins, and 4-quadrant mean error discrepancy ratio. |

### 1.5 Data, Training, Inference & Evaluation (`src/`)
| Path | Key Classes / Functions | Purpose |
| :--- | :--- | :--- |
| `src/data/dataset.py` | `DualDomainDataset`, `create_dataset_splits_from_dir` | Paired spatial RGB + 2D FFT spectral PyTorch dataset loader supporting pre-split (`train`/`val`/`test`) and unified (`real`/`fake`) directories. |
| `src/data/download_dataset.py` | `create_procedural_synthetic_face`, `generate_benchmark_dataset` | Legacy procedural geometric portrait generator. |
| `src/data/generate_realistic_dataset.py` | `generate_variations_from_portraits`, skin/lighting/attribute helpers | Synthesizes 2,400 (`1,600 / 400 / 400`) images from base demo portraits + `whatsapp_test_input.jpg`. |
| `src/train.py` | `precompute_tensors`, `train_one_epoch`, `validate`, `main` | Precomputes frozen EfficientNet-B0 pooled features once, then trains spatial projection + frequency stream + cross-attention + classifier with `LabelSmoothingBCELoss(0.06)` and `AdamW` + `CosineAnnealingLR`. |
| `src/pipeline.py` | `ForensicInferencePipeline` (`analyze_image`) | End-to-end single-image (plus partial multi-face) inference combining face extraction, 2D FFT, ELA, `DualDomainSentinelNet`, heuristic fusion, verdict tiering, Grad-CAM, and stage latencies. |
| `src/evaluation/metrics.py` | `compute_eer`, `evaluate_predictions`, `plot_roc_curve`, `plot_confusion_matrix` | Standard binary metrics, EER threshold finder, ROC & CM plots. |
| `src/evaluation/evaluate_checkpoint.py` | `evaluate_split`, `run_full_evaluation` | Runs sample-by-sample evaluation of `checkpoints/best_model.pth` over `data/test` and `data/val`, writing `checkpoints/evaluation_metrics.json` and 4 PNG curves. |
| `src/evaluation/robustness_bench.py` | `RobustnessBenchmark`, `apply_jpeg_compression`, `apply_gaussian_blur`, `apply_gaussian_noise` | Perturbation sweep helper across JPEG quality factors, blur, and noise. |

### 1.6 Web Application, Reporting & Existing Docs/Tests
| Path | Key Symbols / Contents | Purpose |
| :--- | :--- | :--- |
| `app/app.py` | `load_pipeline`, `load_database`, Streamlit studio UI | Upload / webcam / demo gallery UI, verdict card, multi-face expander, decoupled signals, 5 tabs (Grad-CAM, FFT, ELA, PDF, Performance), plus bottom Performance & SQLite History expanders. |
| `app/performance_dashboard.py` | `load_evaluation_data`, `render_performance_dashboard` | Renders `evaluation_metrics.json`, `training_history.json`, PNG plots, checkpoint SHA-256, spec table, and demo architecture overview table. |
| `app/report_generator.py` | `generate_pdf_report`, `compute_sha256`, `get_checkpoint_sha256` | ReportLab 1-page forensic PDF certificate generator with SHA-256 hashes, evidence table, and 4 artifact thumbnails. |
| `docs/*.md` | `architecture_breakdown.md`, `project_synopsis.md`, `viva_questions.md` | Mathematical derivations, synopsis (still lists older `0.8350 ROC-AUC` / 10 tests), and 15 Q&As (`Q1–Q15`). |
| `tests/test_*.py` | 7 files, 15 unit tests (`test_dashboard`, `test_database`, `test_explainability`, `test_models`, `test_multiface`, `test_pipeline`, `test_report`) | All **15/15 passing in 7.27s** under sandbox env (`HOME` & `MPLCONFIGDIR` set to workspace). |

---

## 2. Current Architecture & Verified Runtime State

### 2.1 Neural Model & Checkpoint Inspection (`checkpoints/best_model.pth`)
- **Checkpoint Format:** Raw `state_dict` (`OrderedDict`, `427` keys, `23,858,550` bytes / `22.75 MB`, SHA-256 `642caf0da86b9280d18648fd173f27db76b0e400f5fbc3472cd41f8346f7e28a`). Strictly loads into `DualDomainSentinelNet(backbone_name="efficientnet_b0", fusion_type="cross_attention")` with `<All keys matched successfully>`.
- **Verified Module Parameter Breakdown** (`sum(p.numel() for p in model.parameters())`):
  - `spatial_stream.features` (EfficientNet-B0 conv layers): **4,007,548** params (frozen during `train.py`)
  - `spatial_stream.proj` (`1280 -> 512` Linear + LayerNorm): **656,896** params
  - `frequency_stream` (`SRMConv` + 4-stage ConvNet + `256 -> 256` FC): **455,968** params
  - `fusion` (`CrossAttentionFusion`, 4 heads, `embed_dim=256`): **724,224** params
  - `classifier` (`256 -> 128 -> 32 -> 1`): **37,377** params
  - **Total Learned Parameters:** **5,882,013** (`1,874,465` trainable under `train.py` frozen-backbone regime)
  - **Non-Param Buffers in `state_dict`** (BatchNorm `running_*` + `num_batches_tracked` + 3 `5x5` SRM kernels): **457,825** elements -> **6,339,838** total `state_dict` elements.
- **Data & Training Mechanics:** `DualDomainDataset` prepares `spatial: (3, 256, 256)` and `spectral: (1, 256, 256)`. `src/train.py` precomputes `(B, 1280)` spatial vectors once (`spatial_flat`), bypassing per-epoch `RandomHorizontalFlip` and `ColorJitter` augmentations on the spatial stream during epochs 1..N.

### 2.2 Installed Runtime Environment (`.venv`)
- **Verified Present:** Python `3.12.6`, `torch 2.15.0` (MPS enabled), `torchvision 0.29.2`, `timm 1.0.30` (installed but unused), `scikit-learn 1.9.1` (`StratifiedGroupKFold` available), `scipy 1.18.1`, `opencv-python-headless 4.14.0.94`, `streamlit 1.66.1`, `reportlab 5.1.0`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `albumentations 2.0.8` (installed, unused), `Pillow 13.0.0`, `pandas 3.1.0`, `pyyaml 6.0.3`, `numpy 2.5.4`, `pytest 9.2.0`.
- **Not Installed (Needed in Upcoming Phases):** `imagehash` (for Phase 2 pHash / near-duplicate detection unless implemented cleanly + added to `requirements.txt`), `psutil` (for Phase 10 CPU/process memory telemetry).

### 2.3 Existing Dataset & Database State
- **Disk Counts:** `train/{real,fake}: 800 / 800`, `val/{real,fake}: 200 / 200`, `test/{real,fake}: 200 / 200` (all `.jpg`, all `(256, 256)` RGB, `0` corrupt files, `0` missing files). `data/demo_samples` contains 8 `(256, 256)` portraits (`4` real, `4` fake); root contains `whatsapp_test_input.jpg` (`225x265`).
- **SQLite Ledger (`data/forensic_history.db`):** `32,768` bytes, 1 table (`forensic_examinations`), **50 rows** (`14` High Authenticity, `18` Inconclusive, `7` Likely Authentic, `7` Likely Manipulated, `4` Unknown).

---

## 3. Missing Functionality (Mapped to Phases 2–12)

1. **Dataset Integrity & Source-Group Leakage Controls (Phase 2):** No SHA-256 / pHash / near-duplicate scanner, no source-group metadata manifest, no `StratifiedGroupKFold` splitting, no `results/dataset_audit.json`.
2. **External Benchmark Evaluation Support (Phase 3):** Zero dataset loaders or evaluators for **FaceForensics++**, **Celeb-DF**, or **DFDC**; no `results/external_benchmark/` outputs (precision-recall curves, FPR/FNR, availability checks).
3. **Architectural Ablation Framework (Phase 4):** No automated ablation runner comparing Spatial-only EfficientNet-B0, Frequency-only SRM+FFT, Spatial+Frequency Concat, Spatial+Frequency Cross-Attention, and Full Model + Existing Loss (`results/ablation_results.json` + comparative charts). `DualDomainSentinelNet` does not yet support `spatial_only` or `frequency_only` modes cleanly.
4. **Confidence Calibration & Validated Uncertainty Policy (Phase 5):** No validation-set temperature scaling, no validation-only threshold optimization / PR threshold curves, no formal 3-tier outcome policy (`AUTHENTIC`, `MANIPULATED`, `UNCERTAIN / NEEDS MANUAL REVIEW`), no **ECE**, **Brier Score**, or **Reliability Diagram**. Heuristic post-processing currently overrides neural outputs.
5. **Video Deepfake Detection Module (Phase 6):** No `src/video/video_analyzer.py`, no MP4/AVI/MOV validation or OpenCV frame extraction/sampling, no frame-level aggregation, suspicious timestamp intervals, interactive Red/Green/Yellow timeline, or frame-level disclaimer.
6. **Explainability Upgrades (Phase 7):** Multi-face pipeline (`len > 1`) omits per-face Grad-CAM overlays, FFT plots, and ELA maps; missing comparison plots between real and suspicious spectral profiles; ELA module/UI lacks explicit forensic limitations disclosure.
7. **Forensic Evidence Integrity & Audit Trail (Phase 8):** Database hashes aligned `256x256` face PNG bytes instead of raw uploaded file bytes; lacks Evidence ID vs Case ID metadata (`file_size`, `dimensions/video_metadata`, `decision_threshold`, `processing_status`), misses an immutable `audit_events` lifecycle table (`upload`, `processing`, `prediction`, `report_generation`), and lacks clear legal chain-of-custody vs software audit log differentiation.
8. **Unified 6-Section Streamlit Studio & Enhanced PDF Reports (Phase 9):** Current UI places everything under a single mode-driven page without dedicated Overview, Image Analysis, Video Analysis, Model Performance, Forensic Evidence, and Model Information sections; PDF generator lacks video forensic reports, threshold metadata, multi-page overflow protection, and accurate domain-fusion text.
9. **Performance Telemetry & Batched Inference (Phase 10):** No batched face/video inference, FPS counter, or CPU/MPS memory telemetry.
10. **Expanded QA Suite & Academic Submission Documentation (Phases 11–12):** Test suite only covers 15 unit checks (none for video, calibration, external loaders, dataset audit, ablation, missing checkpoints, corrupted media). Missing `docs/PROJECT_ARCHITECTURE.md`, `METHODOLOGY.md`, `DATASET_AND_EVALUATION.md`, `TESTING_REPORT.md`, `USER_GUIDE.md`, `TECHNICAL_LIMITATIONS.md`, updated `README.md`, presentation outline, and 30 viva Q&As.

---

## 4. Bugs and Technical Debt

| ID | Severity | Location | Verified Defect / Technical Debt |
| :--- | :--- | :--- | :--- |
| **B1** | **Critical** | `src/pipeline.py:110` vs `src/explainability/spectral_analysis.py:124` | `pipeline.py` queries `spectral_details.get("peak_count", 0)`, whereas `SpectralAnalyzer.analyze_spectral_peaks()` returns `"detected_peaks"` (list) and `"peak_indices"`, never `"peak_count"`. Consequently `peak_count` is always `0` and line 117 (`if peak_count >= 6`) is **dead code**, suppressing spectral deconvolution alerts. |
| **B2** | **High** | `src/pipeline.py:148-156` & `:245-250` | Unvalidated rule-based heuristic blending (`0.65N + 0.18S + 0.17E` for primary face; `0.65N + 0.35E` / `0.70N + 0.30E` for secondary faces) overrides raw `DualDomainSentinelNet` logits and uses inconsistent weights between primary vs secondary faces. |
| **B3** | **High** | `src/pipeline.py:129-135` vs `src/explainability/ela.py:89-97` | `pipeline.py` computes quadrant stds over hardcoded `[:128, :128]` slices (fails/misaligns if image != `256x256`), whereas `ErrorLevelAnalysis.compute_ela_detailed()` computes quadrant **means** (`quad_discrepancy_ratio`). |
| **B4** | **High** | `src/core/face_extractor.py:67-79` | `if not has_eyes and skin_ratio < 0.20: return False` allows false Haar eye triggers on high-contrast shirt text/folds (`minNeighbors=2`) to pass validation even when `skin_ratio` is near `0%` (empirically observed in multi-face screenshots showing `144x144` / `97x97` t-shirt text crops). Also `extract_face()` skips detection on roughly square images (`is_already_square`), while `extract_all_faces()` always detects. |
| **B5** | **Medium** | `src/explainability/gradcam.py:70-76, 121-125` | `score.backward(retain_graph=True)` unnecessarily retains autograd graphs in inference; `target_class` is ignored for 1-logit output `(B, 1)` (`score = output[0, 0]` always computes gradient wrt fake class even if `target_class=0`); `overlay_cam()` does not verify shape compatibility between `original_img_np` and `cam_heatmap`. |
| **B6** | **Medium** | `src/data/dataset.py:66-68, 112-118` | Corrupted images are silently replaced with solid black `(0,0,0)` frames instead of being surfaced/counted; empty `data/test/` silently aliases `test_paths = val_paths`, risking accidental val-on-test inflation. |
| **B7** | **Medium** | `src/train.py:148-199` vs `config/default_config.yaml:32-45` | `train.py` ignores `config["train"]` (`epochs=15`, `batch_size=16`, `label_smoothing=0.05`, `weight_decay=1e-4`), hardcodes `epochs=6`, `batch_size=32`, `LabelSmoothingBCELoss(0.06)`, `weight_decay=1e-3`, and never instantiates `BinaryFocalLoss`. |
| **B8** | **Medium** | `app/performance_dashboard.py:266` & `app/report_generator.py:213-214, 146` | Dashboard claims `"Balanced Binary Focal Loss (alpha = 0.25, gamma = 2.0)"` (mismatch with `train.py` and `BinaryFocalLoss(alpha=0.50)`). PDF report hardcodes outdated fusion text `"0.50N + 0.30S + 0.20E"` and triggers red `verdict_color` whenever `"Manipulated" in verdict_tier or is_fake` (which would color `"Inconclusive"` red if `fake_prob >= 0.50`). |
| **B9** | **Medium** | `src/core/database.py:83-87` & `app/app.py:536-538` | Hashes the post-crop `256x256` PNG rather than the raw uploaded file bytes (cannot verify original file integrity). UI exposes a single-click `Clear Examination History` button that permanently deletes the SQLite ledger without confirmation or audit trail. |
| **B10** | **Low** | Runtime Environment / Caching | `matplotlib` & `streamlit` stall during sandboxed execution unless `HOME` and `MPLCONFIGDIR` point to writable workspace paths. Also `app/performance_dashboard.py` uses `@st.cache_data` without TTL/mtime invalidation on JSON updates. |

---

## 5. Performance Bottlenecks

1. **Unbatched Inference in `evaluate_checkpoint.py` (`evaluate_split`) & `RobustnessBenchmark`:** Processes images one-by-one (`batch_size=1`, repeated PIL -> FFT -> tensor -> `.to(device)` per item), leaving MPS/CUDA GPU vectorization unused.
2. **Repeated Redundant Transforms & FFT/ELA Work (`pipeline.py` & `dataset.py`):** In `pipeline.py`, primary face and secondary faces re-run single-sample forward passes sequentially rather than batching faces `(N_faces, 3, 256, 256)`. Each JPEG `ErrorLevelAnalysis` encode/decode happens in-memory on CPU serially alongside sequential Grad-CAM backward hooks.
3. **Frozen Feature Cache vs Augmentation Lockout (`src/train.py`):** Precomputing `spatial_flat` cuts training speed drastically on MPS, but disables online spatial augmentations (`RandomHorizontalFlip`, `ColorJitter`) and prevents end-to-end fine-tuning experiments during Phase 4 ablation comparisons.
4. **Synchronous UI Matplotlib Rendering (`app/app.py`, `app/performance_dashboard.py`):** Re-creates Matplotlib figures on every Streamlit rerun without figure caching or lightweight frame throttling (crucial once Phase 6 video analysis iterates across dozens/hundreds of frames).

---

## 6. Potential Dataset Leakage & Provenance Findings

> [!CAUTION]
> Phase 1 cryptographic and script inspection confirmed **severe structural and exact duplicate dataset leakage** in the current `data/` directory.

1. **Single Source-Identity Pool Across All Splits:** `src/data/generate_realistic_dataset.py` generates all **2,400** images (`1,600` train, `400` val, `400` test) by cropping and augmenting the **same 5 source portraits** (`4` in `data/demo_samples/authentic_*.jpg` + `whatsapp_test_input.jpg`). Although `test` uses a separate PRNG (`RandomState(999)` vs `42`), the exact same human identities, facial structures, and donor faces appear in `train`, `val`, and `test`.
2. **14 Verified Exact Cross-Split SHA-256 Duplicate Groups (38 Files Total):** Empirical hashing of all 2,400 files in `data/{train,val,test}` revealed:
   - **Groups 01–13 (Cross-Split + Inverted Labels):** 13 distinct images from `data/train/real/` (`real_train_0000.jpg` through `real_train_0100.jpg`) also appear identically (byte-for-byte SHA-256 match) **5 times in `data/val/fake/`** and **11 times in `data/test/fake/`**.
   - **Group 14 (Cross-Split + Mislabelled Synthetic Image):** `data/demo_samples/deepfake_stylegan_checkerboard.jpg` (SHA-256 `3aa67d75...`) appears identically **3 times in `data/val/real/`** (`real_val_0040.jpg`, `0110.jpg`, `0160.jpg`) and **8 times in `data/test/real/`** (`real_test_0022.jpg`, `0045.jpg`, ..., `0192.jpg`).
3. **Training History Provenance:** `checkpoints/training_history.json` (`6` epochs ending `train_acc=0.965`, `val_acc=0.960`, `val_auc=0.983`) does not contain epoch timestamps or run metadata tied to `best_model.pth`, whereas `checkpoints/evaluation_metrics.json` reflects the most recent evaluation of `best_model.pth` on the current `val` (98.00%) and `test` (95.25%, N=400) directories (which include the exact duplicates noted above). No files were altered or deleted in Phase 1; Phase 2 will audit and resolve this with full user visibility.

---

## 7. Model Compatibility Issues

1. **Checkpoint Metadata Absence:** `checkpoints/best_model.pth` stores only tensor weights (`state_dict`), lacking config parameters (`backbone_name`, `fusion_type`, `spatial_dim`, `spectral_dim`), calibrated temperature $T$, or validated decision threshold $\tau$. Loading with a non-default config (`resnet34` or `concat`) would fail unless saved alongside model metadata.
2. **PyTorch `torch.load` Device & `weights_only` Compatibility:** `pipeline.py:52` and `evaluate_checkpoint.py:127` call `torch.load(..., map_location=...)` without `weights_only=True` (triggers future PyTorch security warnings) and assume raw `state_dict` rather than handling both raw `state_dict` and dict-wrapped checkpoints cleanly.
3. **Conditional Forward Signature (`spatial_flat`):** `DualDomainSentinelNet.forward()` branches between `spatial_tensor` and `spatial_flat`, while `GradCAM.generate_cam()` hooks `spatial_stream.target_cam_layer` and falls back to `torch.zeros((b, 1, 256, 256))` if `spectral_tensor is None` — meaning any future ablation model variant (`spatial_only` or `frequency_only`) must cleanly handle or document Grad-CAM applicability.
4. **MPS / CPU Parity & Float Precision:** Apple Silicon MPS executes inference cleanly today (`pytest` passes on CPU; `app.py` runs on MPS), but PyTorch MPS lacks support for certain ops or `float64` FFT edge cases if unguarded; video batching and temperature scaling optimization (`L-BFGS` / `Adam` on CPU logits) must force CPU float32 computation where appropriate.
5. **Dual OpenCV / Package Consistency:** `.venv` has `opencv-python-headless 4.14.0.94`, `scikit-learn 1.9.1` (`StratifiedGroupKFold` supported), while `imagehash` and `psutil` are currently absent and must be accounted for in Phases 2, 10, and `requirements.txt`.

---

## 8. Recommended 12-Phase Implementation Sequence

1. **Phase 1 (Current — Complete):** Complete audit (`docs/PROJECT_AUDIT.md`) + verified baseline backup (`backups/phase1_baseline/`).
2. **Phase 2:** Dataset Integrity & Leakage Audit (`src/data/dataset_audit.py`, SHA-256 + perceptual hashing, source-group clustering, `StratifiedGroupKFold` splitter, `results/dataset_audit.json`, no silent deletions).
3. **Phase 3:** External Dataset Generalization Support (`src/data/external_datasets.py`, independent loaders for FaceForensics++, Celeb-DF, DFDC, `results/external_benchmark/`, zero fabrication / graceful unavailable handling).
4. **Phase 4:** Model Ablation Study (`src/evaluation/ablation_study.py`, flexible fusion/stream modes in `DualDomainSentinelNet`, fair group-safe splits for all 5 experiments -> `results/ablation_results.json` + comparative charts).
5. **Phase 5:** Confidence Calibration & Uncertainty Estimation (`src/evaluation/calibration.py`, temperature scaling on validation logits, validation threshold + PR curve analysis, 3-tier classification `AUTHENTIC` / `MANIPULATED` / `UNCERTAIN / NEEDS MANUAL REVIEW`, ECE + Brier Score + Reliability Diagram; decouple heuristic rules from primary classifier).
6. **Phase 6:** Frame-Level Video Deepfake Detection (`src/video/video_analyzer.py`, MP4/AVI/MOV support, OpenCV sampling, batched face detection & inference, aggregation policy, suspicious intervals, Green/Red/Yellow interactive timeline, explicit frame-level disclaimer).
7. **Phase 7:** Explainability Enhancements (Fix **B1/B3/B4/B5**, per-face Grad-CAM for multi-face inputs, enhanced FFT comparison & anomaly regions, ELA comparisons & limitations, Unified Forensic Evidence Panel).
8. **Phase 8:** Forensic Integrity & SQLite Schema Migration (Raw file SHA-256, Evidence IDs, non-destructive `ALTER TABLE` migration preserving the 50 existing rows + new `audit_events` lifecycle table + legal vs software audit distinction).
9. **Phase 9:** 6-Section Streamlit Dashboard & Upgraded PDF Report Engine (Overview, Image Analysis, Video Analysis, Model Performance, Forensic Evidence, Model Information + image/video PDF certificates with layout guarantees).
10. **Phase 10:** Performance Optimization (Batched face/frame inference, Streamlit caching optimization, device fallback, FPS + CPU/GPU memory telemetry).
11. **Phase 11:** Comprehensive QA & Debugging (Expand unit/integration tests across all 20 required categories + fix any edge-case failures).
12. **Phase 12:** Complete Documentation & Academic Submission Kit (`README.md`, 6 docs in `docs/`, B.Tech project report structure, 12–15 slide outline, 30 viva Q&As).

---

## Appendix A — Phase 1 Baseline Backup Manifest

All critical baseline files were copied cleanly to `backups/phase1_baseline/` prior to any Phase 2+ code modification:
- `backups/phase1_baseline/checkpoints/` (including `best_model.pth`, `evaluation_metrics.json`, `training_history.json`, all `.png` plots, `sample_forensic_report.pdf`)
  - Verified SHA-256 (`checkpoints/best_model.pth` == `backups/.../best_model.pth`): `642caf0da86b9280d18648fd173f27db76b0e400f5fbc3472cd41f8346f7e28a`
- `backups/phase1_baseline/forensic_history.db`
  - Verified SHA-256 (`data/forensic_history.db` == `backups/.../forensic_history.db`): `a4841f05917e983a9deae422cb049fd9a44662143482bfbdf609e1b67bc2ffe1`
- `backups/phase1_baseline/src/`, `app/`, `config/`, `tests/`, `requirements.txt`, `pyproject.toml`, `README.md`
