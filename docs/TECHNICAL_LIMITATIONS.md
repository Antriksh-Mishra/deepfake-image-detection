# Technical Limitations & Scientific Disclosures — Deepfake Forensic Sentinel

**Author:** Antriksh Mishra (Final Year B.Tech CSE Capstone)

Forensic software engineering requires stating empirical system boundaries explicitly alongside capabilities. Below are the verified scientific, algorithmic, and operational limitations of **Deepfake Forensic Sentinel**.

---

## 1. Training Dataset Domain Bound & Unseen-Generator Generalization

- **Bounded Portrait Pool:** While all cross-split identity and SHA-256 duplicate leakage was eliminated (`2,400` images partitioned cleanly across `15` disjoint source groups in [`results/dataset_audit.json`](../results/dataset_audit.json)), those images originate from controlled variations synthesized via [`src/data/generate_realistic_dataset.py`](../src/data/generate_realistic_dataset.py) rather than millions of real-world internet videos.
- **Empirical Generalization Gap:**
  - Validation AUC (`0.7694`) drops slightly on source-disjoint held-out test (`0.7396`).
  - Live probes on [`data/demo_samples/`](../data/demo_samples) with `checkpoints/best_model.pth` confirm accurate separation on `authentic_*.jpg` (`P_cal ~ 0.299`), `deepfake_diffusion_ai_face.jpg` (`P_cal = 0.7223`, `MANIPULATED`), and `deepfake_stylegan_checkerboard.jpg` (`P_cal = 0.7110`, `UNCERTAIN`), whereas unseen local manipulation patterns **`deepfake_boundary_morph.jpg` (`P_cal = 0.2988`)** and **`deepfake_faceswap_manipulation.jpg` (`P_cal = 0.2991`)** produce false negatives (`AUTHENTIC`).
- **Frozen Backbone Trade-Off:** [`src/train.py`](../src/train.py) freezes EfficientNet-B0 convolutional layers and caches pooled vectors once (`spatial_flat`), speeding up Apple Silicon MPS training while disabling online spatial augmentations (`RandomHorizontalFlip` / `ColorJitter`) and high-capacity spatial fine-tuning.
- **External Benchmarks Pending Media Acquisition:** Zero-tuning evaluation loaders exist for **FaceForensics++**, **Celeb-DF**, and **DFDC** ([`src/data/external_datasets.py`](../src/data/external_datasets.py)), but **`0/3`** benchmarks are present locally (`external_benchmark_summary.json`). Out-of-distribution generalization on third-party video generators remains unmeasured until external archives are downloaded.

---

## 2. Calibration & Threshold Shift Under Distribution Shift

- Scalar Temperature Scaling (`T = 2.654271`) and operating threshold (`tau = 0.71`, uncertainty band `[0.7070, 0.7188]`) are strictly fit on validation logits (`data/val`, $N=400$) to protect held-out test integrity (`zero_test_set_tuning = true`).
- Because held-out test logit magnitudes sit slightly lower than validation logits, adopting `tau = 0.71` yields `0` test false positives (`100%` specificity, `100%` precision at binary `0.71`) while increasing binary false negative rate from `48.0%` (at `0.50`) to `61.5%` (at `0.71`), dropping binary test accuracy from `75.50%` to `69.25%`.
- Therefore, operators conducting high-sensitivity screening may override the decision threshold via pipeline/API parameters, recognizing the precision-recall trade-off documented in `results/calibration/pr_threshold_analysis.png`.

---

## 3. Frame-Level Video Scope (`temporal_modeling = False`)

- [`src/video/video_analyzer.py`](../src/video/video_analyzer.py) executes **independent frame-level spatial and 2D FFT inference** across sampled frames combined via 10% symmetric trimmed mean and a sustained multi-frame quorum (`>= max(2, ceil(0.30 * N_face))`).
- **What Is NOT Modeled:**
  - Inter-frame temporal coherence, facial landmark jitter, blink cadence, or physiological rPPG pulse signals;
  - Audio-visual lip-sync or speech waveform deepfake artifacts;
  - Optical-flow warping or 3D spatiotemporal transformers (`3D CNN` / `TimeSformer`).
- Consequently, frame-level consistency can miss purely temporal flickers or audio-only swaps that leave single JPEG-like frames spatially plausible.

---

## 4. Explainability Non-Proof Caveats (Grad-CAM, FFT & ELA)

As enforced inside [`src/explainability/ela.py`](../src/explainability/ela.py) (`ELA_FORENSIC_LIMITATIONS`), [`app/xai_panel.py`](../app/xai_panel.py), and [`app/report_generator.py`](../app/report_generator.py):
1. **Grad-CAM Indicates Classifier Focus, Not Physical Manipulation:** High gradient attribution on ocular or jawline patches reflects which activations drove the network logit; it does not prove localized pixel editing.
2. **Social-Media Recompression Washout (ELA & FFT):** Messaging platforms (WhatsApp, Instagram, Telegram) strip EXIF metadata, resize crops, and re-quantize DCT blocks, erasing genuine splicing residuals while introducing benign quantization harmonics in authentic photos.
3. **Whole-Frame AI Uniformity:** Single-pass diffusion or GAN portraits undergo uniform initial compression across the whole canvas, producing low quadrant ELA mismatch despite being 100% synthetic (empirically seen in `deepfake_diffusion_ai_face.jpg` having low `ela=0.1108`).
4. **Sharp Text & Hair Edges:** Eyeglasses, strands of hair, and specular highlights naturally elevate ELA brightness and mid-band FFT energy in authentic photographs.

---

## 5. Face Detection & Preprocessing Constraints

- [`src/core/face_extractor.py`](../src/core/face_extractor.py) relies on classical OpenCV frontal Haar cascades (`haarcascade_frontalface_default.xml` + `haarcascade_eye.xml`) coupled with YCrCb skin-ratio validation (`>= 14%` on color crops).
- Extreme side profiles (`> 45 deg` yaw), heavy occlusions (masks, sunglasses), or severe backlighting can cause Haar detection misses, triggering either center-crop fallback (images) or `NO FACE DETECTED` skips (video).
- Extreme blur (`Laplacian < 15.0`) or tiny resolutions (`< 80px`) intentionally route to `UNCERTAIN / NEEDS MANUAL REVIEW` rather than forcing an unreliable prediction.

---

## 6. Software Audit Trail vs. Legally Admissible Chain of Custody

As stated in `CHAIN_OF_CUSTODY_DISCLAIMER` ([`src/core/database.py`](../src/core/database.py)):
- The SQLite `forensic_examinations` ledger and SQL-trigger-protected `audit_events` table guarantee **application-level integrity**, exact raw uploaded SHA-256 matching, and chronological lifecycle tracking.
- They do **NOT** constitute standalone legal chain-of-custody evidence under judicial standards, which require external **RFC 3161 Trusted Timestamping Authorities (TSA)**, **PKI/HSM cryptographic signing**, **WORM / hardware write-blocked storage**, verified camera device provenance, and formally attested human custody transfers.

---

## 7. Hardware & CPU Preprocessing Bottleneck

- While warmed batched neural inference on Apple Silicon MPS reaches **`219.9 img/s`** (`batch_size=32`), end-to-end throughput (`~14.3 img/s` on `evaluate_split` `N=40`) is bounded on CPU by serial PIL image decoding, OpenCV Haar detection, 2D NumPy FFT computation, and per-face JPEG ELA re-encoding/Grad-CAM backward hooks.
- Initial PyTorch MPS kernel compilation introduces cold-start warmup overhead on the very first inference call in a fresh process.
