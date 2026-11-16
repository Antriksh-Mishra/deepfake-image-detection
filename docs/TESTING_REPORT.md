# Testing, Performance & QA Verification Report — Deepfake Forensic Sentinel

**Author:** Antriksh Mishra (Final Year B.Tech CSE Capstone)  
**Test Runner:** `pytest 9.2.0` on Python `3.12.6` (Apple Silicon `arm64`, macOS `26.5.2`, PyTorch `2.15.0`)  
**Full Suite Status:** **79 / 79 tests passing (`51.91s`)**

---

## 1. Automated Pytest Verification Matrix (`79` Collected Test Cases Across `15` Modules)

Every test module executes deterministically under headless sandbox settings (`TORCH_HOME`, `HOME`, `MPLCONFIGDIR` pointed to writable workspace directories):

| Test Module | Collected Cases | Verification Scope |
| :--- | :---: | :--- |
| [`tests/test_models.py`](../tests/test_models.py) | 6 | 3 fixed `5x5` SRM kernels shape/values, `SpatialBackbone`, `FrequencyStream`, `CrossAttentionFusion`, `DualDomainSentinelNet` forward shapes across fusion modes, `LabelSmoothingBCELoss` & `BinaryFocalLoss`. |
| [`tests/test_pipeline.py`](../tests/test_pipeline.py) | 1 | End-to-end `ForensicInferencePipeline` inference keys, Grad-CAM, azimuthal FFT, and ELA outputs. |
| [`tests/test_multiface.py`](../tests/test_multiface.py) | 3 | Multi-face extraction ordering, **B4** non-skin clothing/text false-positive rejection, per-face Grad-CAM + FFT + ELA generation across multiple faces. |
| [`tests/test_explainability.py`](../tests/test_explainability.py) | 4 | **B1** spectral `detected_peaks` fix, **B3** dynamic ELA quadrant means & `Q=95/90/75` recompression comparison, **B5** Grad-CAM negative/positive target class & shape check, `ELA_FORENSIC_LIMITATIONS` presence. |
| [`tests/test_dataset_audit.py`](../tests/test_dataset_audit.py) | 5 | Raw SHA-256 exact duplicates, 64-bit 2D DCT pHash near-duplicate clustering, `StratifiedGroupKFold` zero-leakage guarantees, corrupted/missing image handling, live `data/` dataset audit verification (`0` leaked groups). |
| [`tests/test_external_benchmark.py`](../tests/test_external_benchmark.py) | 6 | Independent synthetic directory loaders & evaluators for FaceForensics++ (`c23`/`c40`), Celeb-DF, DFDC (`metadata.json`/`labels.csv`), corrupted frame tolerance, and verified `external_benchmark_summary.json` honesty (`available: false` when absent). |
| [`tests/test_ablation.py`](../tests/test_ablation.py) | 5 | `DualDomainSentinelNet` support for `spatial_only`, `frequency_only`, `concat`, `cross_attention`, group-disjoint ablation execution, and schema verification of `results/ablation_results.json`. |
| [`tests/test_calibration.py`](../tests/test_calibration.py) | 8 | Temperature scaling NLL monotonicity, `ECE` (`positive_class` & `confidence`), Brier score, validation-only Youden's J threshold selection, 3-outcome band & sharpness/dimension quality guards, live `calibration_results.json` consistency. |
| [`tests/test_video.py`](../tests/test_video.py) | 7 | MP4/AVI/MOV validation & corrupt stream rejection, single-frame false-positive immunity (quorum rule), suspicious interval grouping, parameterized end-to-end video analysis (`fps`, `interval`, `uniform`), and zero-face safe handling. |
| [`tests/test_database.py`](../tests/test_database.py) | 4 | Schema v2 initialization & non-destructive legacy `ALTER TABLE` migration preserving prior rows, raw SHA-256 vs crop SHA-256 tamper detection, `audit_events` immutability trigger enforcement (`UPDATE`/`DELETE` aborted), two-step soft archive. |
| [`tests/test_report.py`](../tests/test_report.py) | 3 | **B8** amber verdict theme priority on borderline/uncertain cases, multi-page ReportLab Image PDF certificate compilation (with raw SHA-256 & `audit_events`), multi-page Video PDF certificate compilation (`temporal_modeling=False` disclosure). |
| [`tests/test_dashboard.py`](../tests/test_dashboard.py) | 1 | `load_evaluation_data()` schema & artifact loading integrity. |
| [`tests/test_performance.py`](../tests/test_performance.py) | 6 | `get_runtime_memory_telemetry()` (`psutil`/`resource` fallback), `safe_batched_forward()` automatic MPS/CUDA `RuntimeError -> CPU` fallback, multi-face batched parity (`1e-5`), batched `evaluate_split` + `RobustnessBenchmark` parity, **B10** `st_mtime_ns`/`st_size` Streamlit cache invalidation, and video telemetry completeness. |
| [`tests/test_end_to_end_qa.py`](../tests/test_end_to_end_qa.py) | 11 | Live calibrated authentic vs manipulated vs uncertain portrait routing, explicit threshold overrides, edge image modes/sizes (`L`, `RGBA`, `CMYK`, `1x1`, `60x60` triggering resolution/sharpness guards), 3-face composite -> DB -> real PDF, real video -> DB -> multi-page PDF, missing checkpoint/corrupt dataset/KeyError fallbacks, full **Streamlit `AppTest` 6-section interactive UI workflow** (isolated via `DFAKE_DB_PATH` & `DFAKE_REPORT_PATH`), and production checkpoint/DB integrity guards. |
| [`tests/test_sneaker_and_nonface_qa.py`](../tests/test_sneaker_and_nonface_qa.py) | 6 | Sneaker / product object localization contour extraction, non-face Grad-CAM structural zone labels (no nasolabial/ocular false attribution), strict $P(\text{Fake}) > 50\%$ verdict invariant ($P=70.25\%$ can never be REAL), container-aware ELA (PNG lossless handling), Spatial SRM analyzer ($F_1, F_2, F_3$ high-pass energy & block CV), and zero-gap latency telemetry parity. |
| **Total** | **79** | **79 Passed, 0 Failed** |

---

## 2. Phase 10 Batched Performance & Numerical Parity Benchmarks

Measured on Apple Silicon (`mps`) vs `cpu` in Phase 10 with `checkpoints/best_model.pth`:

| Benchmark Workload | Configuration | Elapsed Time | Throughput | Comparison / Parity |
| :--- | :--- | :---: | :---: | :--- |
| **Neural Forward (`N=64` cached tensors)** | **MPS `batch_size=32` (warmed)** | **291.1 ms** | **219.9 img/s** | **`1.60x` vs MPS `bs=1`; `26.4x` vs CPU `bs=32`** |
| Neural Forward (`N=64` cached tensors) | MPS `batch_size=1` (warmed) | 464.6 ms | 137.7 img/s | Unbatched MPS baseline |
| Neural Forward (`N=64` cached tensors) | CPU `batch_size=32` | 7,681.5 ms | 8.3 img/s | CPU reference baseline |
| **End-to-End `evaluate_split` (`N=40`)** | MPS `batch_size=1` vs `batch_size=16` | ~2,790 ms | ~14.3 img/s | Dominant bottleneck: CPU PIL/2D FFT prep; probability max diff = **`2.89e-06`** |

---

## 3. Defects Resolved During Engineering & QA Lifecycle

| Defect ID | Description | Verified Fix |
| :--- | :--- | :--- |
| **B1** | `pipeline.py` queried nonexistent key `"peak_count"` instead of `len(detected_peaks)`, making deconvolution peak alerts dead code. | Updated to `len(spectral_details.get("detected_peaks", []))` (`tests/test_explainability.py`). |
| **B2** | Legacy heuristic score blending (`0.65N + 0.18S + 0.17E`) overrode neural logits inconsistently across faces. | Removed heuristic override completely (`heuristic_override_applied = False`); temperature-scaled neural probability governs decisions (`tests/test_calibration.py`). |
| **B3** | `pipeline.py` computed ELA quadrant stds on hardcoded `[:128, :128]` slices. | Unified around dynamic dimension-aware `ErrorLevelAnalysis.compute_ela_detailed()` quadrant means & `16x16` block CV (`tests/test_explainability.py`). |
| **B4** | Haar eye cascade triggers passed non-skin clothing folds/text (`0%` skin) when eye Haar misfired. | Enforced `skin_ratio >= 0.14` unconditional minimum on color crops + `minNeighbors=3` eye gate (`tests/test_multiface.py`). |
| **B5** | `GradCAM.generate_cam()` used `retain_graph=True` and ignored `target_class` on single-logit `(B, 1)` output. | Removed `retain_graph=True`, computed `+logit` / `-logit` per `target_class`, added `overlay_cam()` shape checks (`tests/test_explainability.py`). |
| **B6** | `DualDomainDataset` silently swapped corrupt images or aliased empty test sets to val. | Tracks `corrupted_paths`, exposes `is_corrupt`, raises explicitly instead of silent val-on-test leakage (`tests/test_dataset_audit.py`). |
| **B8** | PDF generator colored `"Inconclusive"` verdicts red and printed outdated fusion equation `"0.50N + 0.30S + 0.20E"`. | `_resolve_verdict_theme()` checks uncertain/no-face before `is_fake` (Amber `#d97706`) and documents calibrated dual-domain fusion (`tests/test_report.py`). |
| **B9** | DB hashed post-crop PNG instead of raw upload bytes; UI delete wiped SQLite history outright. | Added `raw_file_sha256`, append-only `audit_events` with SQLite `BEFORE UPDATE/DELETE` abort triggers, and two-step soft archive (`tests/test_database.py`). |
| **B10** | `@st.cache_data` in `performance_dashboard.py` failed to invalidate immediately when evaluation JSON files updated. | Replaced with `get_evaluation_artifacts_signature()` keying `@st.cache_data` on `(path, st_mtime_ns, st_size)` (`tests/test_performance.py`). |
| **B11** | False-negative classification on elevated probability ($P=70.25\%$ with $\tau=0.71$, $u_{low}=0.707$ resolving to AUTHENTIC). | Enforced strict mathematical invariant: $P(\text{Fake}) > 50\%$ can never resolve to `REAL` / `AUTHENTIC`; ambiguous/margin cases resolve to `UNCERTAIN` (`tests/test_calibration.py`, `tests/test_sneaker_and_nonface_qa.py`). |
| **B12** | Non-face images (sneakers, products, vehicles, artwork) falsely attributed facial regions ("nasolabial / mid-facial", "ocular"). | Added `detect_object_bounds` salient contour localization, `classify_image_context`, and structural regional zones (`Upper Perimeter`, `Object Core`, `Base Boundary`) (`src/core/face_extractor.py`, `src/explainability/gradcam.py`). |
| **B13** | ELA falsely claimed JPEG compression history on lossless PNG / WebP uploads. | Implemented container-format awareness in `ErrorLevelAnalysis`, dynamically detecting lossless sources and qualifying compression difference metrics as exploratory (`src/explainability/ela.py`). |
| **B14** | SRM noise residuals were only extracted inside the frequency stream without standalone forensic explainability. | Created `SpatialSRMAnalyzer` providing standalone $F_1, F_2, F_3$ high-pass residual maps, reflection padding, residual energy, 16x16 block CV noise inconsistency, and thermal visualization (`src/explainability/srm_analysis.py`). |
| **B15** | Component latencies summed to 373.2 ms vs 410.5 ms total (37.3 ms unmeasured gap); throughput labeled "face/second". | Added explicit timing boundaries for preprocessing, neural inference, explainability, and postprocessing ensuring exact sum parity within 0.1 ms; standardized throughput to `img/s` (`src/pipeline.py`, `app/app.py`). |
| **Phase 11 Thread Bug** | Streamlit `AppTest` under pytest crashed on macOS GUI backend (`RuntimeError: Cannot create a GUI FigureManager outside the main thread using the MacOS backend`). | Enforced non-interactive `matplotlib.use("Agg")` (`force=True` in `app/app.py`) across `app/`, `src/evaluation/`, and `tests/test_video.py`. |

---

## 4. Re-Running Verification Yourself

```bash
# Full 73-test automated QA suite (CPU/MPS safe, isolated tmp DBs & reports)
TORCH_HOME=/Users/antrikshmishra/.cache/torch \
HOME=/Users/antrikshmishra/Desktop/dfake \
MPLCONFIGDIR=/Users/antrikshmishra/Desktop/dfake/.matplotlib \
.venv/bin/pytest tests/ -q

# Verify production checkpoint & clean database integrity remain untouched
shasum -a 256 checkpoints/best_model.pth backups/phase1_baseline/checkpoints/best_model.pth
```
