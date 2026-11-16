# Dataset Integrity & Empirical Evaluation — Deepfake Forensic Sentinel

**Author:** Antriksh Mishra (Final Year B.Tech CSE Capstone)  
**Verified Artifacts:** [`results/dataset_audit.json`](../results/dataset_audit.json), [`checkpoints/evaluation_metrics.json`](../checkpoints/evaluation_metrics.json), [`checkpoints/training_history.json`](../checkpoints/training_history.json), [`results/calibration/calibration_results.json`](../results/calibration/calibration_results.json), [`results/ablation_results.json`](../results/ablation_results.json), [`results/external_benchmark/external_benchmark_summary.json`](../results/external_benchmark/external_benchmark_summary.json)

---

## 1. Dataset Integrity Audit & Source-Disjoint Resolution

During Phase 1 technical auditing (documented in [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md)), inspection of the initial workspace dataset revealed structural source-portrait sharing and 14 exact cross-split SHA-256 duplicate groups inflating un-audited metrics.

To establish rigorous scientific hygiene, the dataset pipeline was upgraded via [`src/data/generate_realistic_dataset.py`](../src/data/generate_realistic_dataset.py) (`source_split_mode="disjoint"`) and audited read-only by [`src/data/dataset_audit.py`](../src/data/dataset_audit.py).

### Verified Audit Summary (`results/dataset_audit.json`)

| Integrity Check | Verified Result |
| :--- | :--- |
| **Total Images / Valid RGB** | `2,400` / `2,400` (`1,200` Authentic Real, `1,200` Manipulated Fake) |
| **Split Distribution** | `train`: 1,600 (`800/800`) \| `val`: 400 (`200/200`) \| `test`: 400 (`200/200`) |
| **Resolution & Color Mode** | `256x256` (`2,400/2,400`), `RGB` (`2,400/2,400`) |
| **Corrupt / Missing Files** | `0` corrupt, `0` missing |
| **Exact SHA-256 Duplicates** | `2,400` unique SHA-256 digests (`0` duplicate groups, `0` cross-split exact copies) |
| **Source Groups (`source_groups.json`)** | `15` explicit source groups (`9` train, `3` val, `3` test) — **`0` cross-split leaked groups** |
| **64-bit 2D DCT pHash (`threshold <= 4`)** | `1,653` near-duplicate images **strictly within-split** (synthetic variations of same base pool), **`0` cross-split near duplicates** (median cross-split Hamming distance: `16.0` bits) |
| **Controlled Ablation Manifest (`StratifiedGroupKFold`)** | `results/group_split_manifest.json`: `1,426` train (9 groups) / `488` val (3 groups) / `486` test (3 groups) |

---

## 2. Production Model Training & Uncalibrated Baseline Evaluation

- **Checkpoint File:** [`checkpoints/best_model.pth`](../checkpoints/best_model.pth) (SHA-256: `e8c3f62ac628323f0a3daee52923eee611da7850bf7c05c5481dc44ff49cde00`, preserved bit-identical to [`backups/phase1_baseline/checkpoints/best_model.pth`](../backups/phase1_baseline/checkpoints/best_model.pth)).
- **Training History (`checkpoints/training_history.json`, 6 epochs):**
  - Epoch 1: `train_loss=0.4189`, `train_acc=88.63%`, `val_loss=0.9400`, `val_acc=59.75%`, `val_auc=0.7041`
  - Epoch 2 (Best Val AUC): `train_loss=0.2664`, `train_acc=98.56%`, `val_loss=0.8278`, `val_acc=69.00%`, `val_auc=0.7694`, `val_f1=0.7130`
  - Epoch 6 (Final): `train_loss=0.2336`, `train_acc=100.00%`, `val_loss=0.9456`, `val_acc=67.25%`, `val_auc=0.7541`, `val_f1=0.6975`

*Interpretation:* Under a small `15`-group synthetic portrait domain, frozen EfficientNet-B0 representations reach `100%` training fit quickly while source-disjoint validation plateaus around `~0.75–0.77` AUC-ROC—highlighting why source-disjoint evaluation and validation calibration are mandatory.

### Baseline Metrics (`checkpoints/evaluation_metrics.json`, Threshold = `0.50`, Uncalibrated)

| Metric | Held-Out Test (`N=400`, 200R / 200F) | Validation (`N=400`, 200R / 200F) |
| :--- | :---: | :---: |
| **Accuracy** | **75.50%** | 69.00% |
| **Precision** | 0.9811 | 0.6638 |
| **Recall (Sensitivity)** | 0.5200 | 0.7700 |
| **Specificity** | 0.9900 | 0.6100 |
| **F1-Score** | 0.6797 | 0.7130 |
| **ROC-AUC** | 0.7396 | 0.7694 |
| **PR-AUC** | 0.8295 | 0.8326 |
| **False Positive Rate (FPR)** | 1.00% (`2/200`) | 39.00% (`78/200`) |
| **False Negative Rate (FNR)** | 48.00% (`96/200`) | 23.00% (`46/200`) |
| **Equal Error Rate (EER)** | 32.00% | 23.00% |
| **Confusion Matrix (`[[TN, FP], [FN, TP]]`)** | `[[198, 2], [96, 104]]` | `[[122, 78], [46, 154]]` |

---

## 3. Validation-Only Calibration & 3-Outcome Policy (`calibration_results.json`)

Executed by `src/evaluation/calibration.py` strictly fitting all parameters on `data/val` (`zero_test_set_tuning = true`, `bayesian_heuristic_override_removed = true`):

- **Fitted Scalar Temperature:** **`T = 2.654271`**
- **Validation Operating Threshold (Youden's J & Best-F1):** **`tau = 0.71`**
- **Validation 3-Outcome Uncertainty Band:** **`[0.7070, 0.7188]`**

### Calibration Error & Out-of-Sample Effect

| Split | State | Threshold | NLL | Brier | ECE (Pos-Class) | ECE (Confidence) | MCE (Pos) | Accuracy | Prec / Rec / F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Uncalibrated (`T=1.0`) | `0.50` | 0.7728 | 0.2522 | 0.2364 | 0.2098 | 0.7402 | 69.00% | 0.664 / 0.770 / 0.713 |
| **Validation** | Calibrated (`T=2.6543`) | **`0.71`** | **0.6097** | **0.2098** | **0.1142** | **0.0484** | **0.5656** | **77.25%** | **0.785 / 0.750 / 0.767** |
| **Held-Out Test** | Uncalibrated (`T=1.0`) | `0.50` | 0.6247 | 0.2011 | 0.1867 | 0.1488 | 0.7531 | 75.50% | 0.981 / 0.520 / 0.680 |
| **Held-Out Test** | Calibrated (`T=2.6543`) | `0.71` | **0.5571** | **0.1850** | **0.0983** | **0.0655** | **0.4430** | 69.25% | **1.000** / 0.385 / 0.556 |

### Three-Outcome Selective Evaluation (`[0.7070, 0.7188]`)

| Split (`N=400`) | `AUTHENTIC` (<0.707) | `MANIPULATED` (>0.7188) | `UNCERTAIN` (`[0.707, 0.7188]`) | Coverage | Selective Acc | In-Band Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | 204 | 131 | 65 (`16.25%`) | **83.75%** | **82.69%** | 49.23% (near-coin-flip ambiguity!) |
| **Held-Out Test** | 319 | 51 (`51 TP, 0 FP`) | 30 (`7.50%`) | 92.50% | 67.84% | 86.67% |

*Scientific Analysis:* Temperature scaling cuts **held-out test NLL by 10.8% (`0.6247 -> 0.5571`), Brier by 8.0%, and positive-class ECE by 47.4% (`0.1867 -> 0.0983`)** without altering ROC-AUC (`0.7396`). On validation, cases falling inside `[0.7070, 0.7188]` had `49.23%` accuracy—confirming genuine epistemic uncertainty and boosting selective accuracy on decided cases from `77.25%` to `82.69%`. On held-out test, out-of-sample logit shift causes `tau=0.71` to prioritize zero false positives (`100%` precision, `CM=[[200, 0], [123, 77]]`) at the expense of higher FNR (`61.5%`).

Generated diagnostic figures: [`results/calibration/reliability_diagram.png`](../results/calibration/reliability_diagram.png), [`results/calibration/confidence_distribution.png`](../results/calibration/confidence_distribution.png), [`results/calibration/pr_threshold_analysis.png`](../results/calibration/pr_threshold_analysis.png).

---

## 4. Controlled 5-Experiment Ablation Study (`results/ablation_results.json`)

Protocol (`1,426` train / `488` val / `486` test on `group_split_manifest.json`, `8` epochs, `batch_size=32`, `lr=3e-4`, `seed=42`, Apple Silicon `MPS`, single-image `bs=1` latency):

| Exp | Architecture / Fusion | Loss Function | Total Params | Trainable | Acc (%) | Prec | Rec | F1 | ROC-AUC | PR-AUC | EER (%) | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1** | Spatial-Only (`EfficientNet-B0`) | `BCEWithLogitsLoss` | 4,734,589 | 727,041 | 59.05 | 0.552 | 0.955 | 0.700 | 0.8251 | 0.8188 | 25.51 | **9.27 ms** |
| **Exp 2** | Frequency-Only (`SRM + FFT`) | `BCEWithLogitsLoss` | **493,345** | 493,345 | **80.04** | **0.880** | 0.695 | 0.777 | **0.9092** | **0.9102** | 20.99 | 12.19 ms |
| **Exp 3** | Spatial + Freq (`Concat`) | `BCEWithLogitsLoss` | 5,223,325 | 1,215,777 | 78.60 | 0.762 | 0.831 | 0.795 | 0.8840 | 0.8809 | 21.81 | 38.33 ms |
| **Exp 4** | Spatial + Freq (`Cross-Attention`) | `BCEWithLogitsLoss` | 5,882,013 | 1,874,465 | 76.54 | 0.719 | 0.872 | 0.788 | 0.8901 | 0.9036 | 23.46 | 17.38 ms |
| **Exp 5** | Full Model + Reg Loss | `LabelSmoothingBCE(0.06)` | 5,882,013 | 1,874,465 | 79.84 | 0.736 | **0.930** | **0.822** | 0.8752 | 0.8466 | **19.55** | 10.11 ms |

### Empirical Ablation Takeaways
1. **Why Spatial-Only Fails Under Source-Disjoint Generalization (`59.05%` Acc, `77.4%` FPR):** Frozen ImageNet RGB features overfit to identity/lighting differences across the small donor pool and over-trigger on unseen real identities (`188/243` false positives).
2. **Frequency + SRM Drives Spectral Separation (`0.9092` AUC, `80.04%` Acc):** Because synthetic periodic deconvolution/noise artifacts live in 2D FFT residuals rather than RGB appearance, the compact 493K-parameter Frequency Stream achieves the highest ROC-AUC (`0.9092`) and PR-AUC (`0.9102`).
3. **Full Cross-Attention + Label Smoothing Yields Best F1 & Lowest EER (`0.822` F1, `19.55%` EER):** Regularizing the cross-attention dual-domain model cuts EER from `23.46%` (Exp 4) and `25.51%` (Exp 1) to `19.55%` while improving F1 from `0.700` to `0.822` at `10.11 ms/image`.

---

## 5. Live Calibrated Demo Portrait Probes (`data/demo_samples/`, CPU `T=2.6543, tau=0.71`)

Verified directly against `checkpoints/best_model.pth`:

| Sample Filename | Raw Logit | Uncal $P$ | Calibrated $P$ | 3-Outcome Verdict | Interpretation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `authentic_female_portrait.jpg` | -2.2601 | 0.0945 | 0.2991 | `AUTHENTIC` | True Negative |
| `authentic_male_portrait.jpg` | -2.2606 | 0.0944 | 0.2991 | `AUTHENTIC` | True Negative |
| `authentic_studio_photo_1.jpg` | -2.2568 | 0.0948 | 0.2994 | `AUTHENTIC` | True Negative |
| `authentic_studio_photo_2.jpg` | -2.2626 | 0.0943 | 0.2989 | `AUTHENTIC` | True Negative |
| `deepfake_diffusion_ai_face.jpg` | +2.5370 | 0.9267 | **0.7223** | `MANIPULATED` | True Positive (`> 0.7188`) |
| `deepfake_stylegan_checkerboard.jpg` | +2.3889 | 0.9160 | **0.7110** | `UNCERTAIN / NEEDS MANUAL REVIEW` | Borderline inside `[0.7070, 0.7188]` |
| `deepfake_boundary_morph.jpg` | -2.2636 | 0.0942 | 0.2988 | `AUTHENTIC` | **False Negative** (unseen blending pattern) |
| `deepfake_faceswap_manipulation.jpg` | -2.2599 | 0.0945 | 0.2991 | `AUTHENTIC` | **False Negative** (unseen faceswap pattern) |

---

## 6. External Generalization Benchmarks (`results/external_benchmark/`)

[`src/data/external_datasets.py`](../src/data/external_datasets.py) and [`src/evaluation/external_benchmark.py`](../src/evaluation/external_benchmark.py) implement zero-tuning evaluators (`model.eval()`, fixed `threshold=0.50` unless configured, uniform clip averaging for videos, dataset-specific ROC/PR/CM plotting) for three public benchmarks:

| Dataset | Expected Directory / Env Var | Verified Local Status (`external_benchmark_summary.json`) |
| :--- | :--- | :--- |
| **FaceForensics++ (FF++)** | `data/external/FaceForensics++` (`FFPP_DIR`) | **`unavailable`** (`0/3` locally installed; zero fabricated scores) |
| **Celeb-DF (v1/v2)** | `data/external/Celeb-DF` (`CELEBDF_DIR`) | **`unavailable`** (`0/3` locally installed; zero fabricated scores) |
| **DFDC** | `data/external/DFDC` (`DFDC_DIR`) | **`unavailable`** (`0/3` locally installed; zero fabricated scores) |

Synthetic directory-structure integration tests confirm end-to-end loading, video frame sampling, ROC/PR/CM generation, and honest unavailability handling in [`tests/test_external_benchmark.py`](../tests/test_external_benchmark.py).

---

## 7. Multi-Category Out-of-Sample Empirical Evaluation (`results/category_evaluation.json`)

To evaluate the generalization boundaries of `checkpoints/best_model.pth` across diverse visual domains (including non-face objects, commercial products, animals, and digital artifacts), [`src/evaluation/category_eval.py`](../src/evaluation/category_eval.py) audited 11 distinct media categories:

| Category | Samples ($N$) | Ground Truth | Accuracy | Precision | Recall | F1 | Brier Score | ECE | Mean Calibrated $P(\text{Fake})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Photographs** | 20 | Authentic | **100.0%** | N/A ($0.0$) | N/A ($0.0$) | N/A ($0.0$) | **0.0909** | 0.3013 | 0.3013 |
| **AI-Generated Images** | 20 | Manipulated | 25.0% | 1.0000 | 0.2500 | 0.4000 | 0.3882 | 0.5961 | 0.4039 |
| **Diffusion AI Faces** | 1 | Manipulated | **100.0%** | 1.0000 | 1.0000 | **1.0000** | **0.0771** | 0.2777 | 0.7223 |
| **StyleGAN Checkerboard** | 1 | Manipulated | **100.0%** | 1.0000 | 1.0000 | **1.0000** | **0.0835** | 0.2890 | 0.7110 |
| **Face-Swapped / Morphed** | 2 | Manipulated | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0.4914 | 0.7010 | 0.2990 |
| **Product Photographs** | 20 | Balanced ($10\text{R}/10\text{F}$) | 50.0% | 0.5000 | 1.0000 | 0.6667 | 0.2985 | 0.2202 | 0.7202 |
| **Animal Images** | 16 | Balanced ($8\text{R}/8\text{F}$) | 50.0% | 0.5000 | 1.0000 | 0.6667 | 0.2990 | 0.2208 | 0.7208 |
| **Landscapes & Scenes** | 16 | Balanced ($8\text{R}/8\text{F}$) | 50.0% | 0.5000 | 1.0000 | 0.6667 | 0.2604 | 0.1022 | 0.6022 |
| **Compressed Media** | 18 | Balanced ($9\text{R}/9\text{F}$) | 50.0% | 0.0000 | 0.0000 | 0.0000 | 0.2905 | 0.2674 | 0.2994 |
| **Resized Media** | 12 | Balanced ($6\text{R}/6\text{F}$) | 50.0% | 0.0000 | 0.0000 | 0.0000 | 0.2904 | 0.2007 | 0.2993 |
| **Digital Screenshots** | 12 | Balanced ($6\text{R}/6\text{F}$) | 50.0% | 0.5000 | 1.0000 | 0.6667 | 0.2971 | 0.2171 | 0.7171 |

### Scientific Conclusions from Multi-Category Evaluation:
1. **Domain Sensitivity:** Models trained primarily on synthetic facial portraits demonstrate elevated sensitivity to high-frequency edge gradients found in non-face products, sneakers, and screenshots (mean $P(\text{Fake}) \approx 0.71\text{--}0.72$), highlighting why decoupled auxiliary evidence (SRM noise consistency, 2D FFT optical decay, ELA) and the `UNCERTAIN` verdict state are essential forensic safeguards.
2. **Lossless vs Lossy Artifacts:** Heavy JPEG recompression smooths high-frequency SRM residuals, pushing predictions toward authentic baselines ($P \approx 0.2994$). ELA and Spatial SRM residuals must be interpreted in light of the image container format.
