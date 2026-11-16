# B.Tech Final Year Capstone Submission Kit — Deepfake Forensic Sentinel

**Project Title:** Deepfake Forensic Sentinel: Calibrated Dual-Domain Image & Frame-Level Video Deepfake Detection with Explainable AI and Software Forensic Audit Trails  
**Student Name:** Antriksh Mishra  
**Degree Program:** Bachelor of Technology (B.Tech) — Computer Science & Engineering  

---

## Part A — Complete B.Tech Project Report Chapter Structure

1. **Front Matter:** Title Page, Certificate, Declaration, Abstract (dual-domain hypothesis, calibrated `75.50%` test acc / `0.7396` AUC @ `0.50`, `T=2.6543` calibration, frame-level video quorum, `73/73` QA), Acknowledgement, Table of Contents, List of Figures (confusion matrices, ROC curves, reliability diagram, PR curve, ablation charts, video timeline, XAI outputs), List of Tables & Abbreviations.
2. **Chapter 1: Introduction:** Generative media risks (StyleGAN / Diffusion / FaceSwap), generalization gap of RGB-only CNNs, problem statement, 5 project objectives, report organization.
3. **Chapter 2: Literature Review:** Spatial detectors (MesoNet, Xception, EfficientNet), frequency/SRM forensics (2D DFT azimuthal analysis, SRM steganalysis), confidence calibration (Guo et al. temperature scaling, selective abstention), XAI interpretability vs non-proof standards (SWGDE/ENFSI), video forensics vs temporal modeling scope.
4. **Chapter 3: Dataset Engineering, Leakage Audit & System Architecture:** Source-disjoint data synthesis (`2,400` images, `15` groups), SHA-256 & 64-bit 2D DCT pHash audit (`0` leaked groups), `StratifiedGroupKFold` manifest (`1,426 / 488 / 486`), end-to-end 4-stage pipeline architecture (`FaceExtractor -> DualDomainSentinelNet -> Validation Calibration -> Decoupled XAI + SQLite v2`). (Grounded in [`docs/PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md)).
5. **Chapter 4: Mathematical & Algorithmic Methodology:** 2D DFT log-magnitude & radial `1.69 * log1p(r)` baseline, 3 fixed `5x5` SRM kernels, 4-head Cross-Attention equations, `LabelSmoothingBCELoss(0.06)` vs `BinaryFocalLoss`, temperature scaling L-BFGS & Youden's J thresholding, 3-outcome band & sharpness/size guards, negative/positive-logit Grad-CAM, multi-quality ELA, 10% trimmed mean + sustained video quorum. (Grounded in [`docs/METHODOLOGY.md`](METHODOLOGY.md)).
6. **Chapter 5: Implementation Details:** Hardware/software stack (Python `3.12.6`, PyTorch `2.15.0` MPS/CPU fallback, Streamlit `1.67.0`, ReportLab `5.1.0`), 6-section Streamlit Studio (`app/app.py`), SQLite `audit_events` triggers (`src/core/database.py`), multi-page PDF certificate layout (`app/report_generator.py`), batched inference optimization (`src/core/utils.py`).
7. **Chapter 6: Experimental Results, Ablation & Testing:** Held-out test/val baselines (`evaluation_metrics.json`), calibration before/after NLL/ECE/Brier tables (`calibration_results.json`), 5-experiment ablation analysis (`ablation_results.json`), external benchmark loader readiness & honest unavailability (`external_benchmark_summary.json`), MPS vs CPU batching speedup (`219.9 img/s` vs `8.3 img/s`), `73/73` QA & Streamlit `AppTest` verification. (Grounded in [`docs/DATASET_AND_EVALUATION.md`](DATASET_AND_EVALUATION.md) & [`docs/TESTING_REPORT.md`](TESTING_REPORT.md)).
8. **Chapter 7: Technical Limitations, Ethics & Legal Scope:** Synthetic training domain bounds, unseen-generator false negatives (`deepfake_boundary_morph`, `deepfake_faceswap`), `temporal_modeling=False` scope, XAI recompression washout caveats, software audit logs vs RFC-3161 legal chain of custody. (Grounded in [`docs/TECHNICAL_LIMITATIONS.md`](TECHNICAL_LIMITATIONS.md)).
9. **Chapter 8: Conclusion & Future Scope:** Summary of verified contributions; future external FF++/Celeb-DF/DFDC benchmarking, end-to-end unfreezing, and inter-frame/audio-visual modeling.
10. **References & Appendices:** Appendix A (CLI & Studio User Guide), Appendix B (30 Viva Questions), Appendix C (Sample PDF Forensic Certificate & Verification Logs).

---

## Part B — 14-Slide Capstone Defense Presentation Outline

| Slide | Slide Title | Key Bullet Points & Visual Evidence to Embed |
| :---: | :--- | :--- |
| **1** | Title Slide | **Deepfake Forensic Sentinel**: Dual-Domain Detection, Calibration & Explainable Forensics — *Antriksh Mishra (Final Year B.Tech CSE)*. |
| **2** | Problem Statement & Motivation | Why RGB-only CNNs collapse (`59.05%` Acc / `77.4%` FPR in our Spatial-Only ablation) and why uncalibrated black-box binaries fail forensic standards. |
| **3** | Key Project Objectives | Dual-domain Spatial+FFT fusion; Leakage-free evaluation; Validation-only Temperature Scaling & 3-outcome policy; Frame-level video & audit trail. |
| **4** | System Architecture Overview | End-to-end architecture flow diagram ([`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md)): Raw SHA-256 -> `FaceExtractor` -> `DualDomainSentinelNet` (`5.88M` params) -> 3-Outcome Policy -> XAI & PDF. |
| **5** | Spatial + Frequency + Cross-Attention Design | EfficientNet-B0 (`512`-d) + 3 Fixed `5x5` SRM Kernels over 2D FFT (`256`-d) + 4-Head Cross-Attention Fusion + `LabelSmoothingBCELoss(0.06)`. |
| **6** | Dataset Hygiene & Leakage Elimination | Detecting Phase 1 cross-split duplicates -> Rebuilding `2,400` images across `15` disjoint source groups (`0` exact duplicates, `0` leaked groups via 64-bit DCT pHash & `StratifiedGroupKFold`). |
| **7** | Controlled 5-Experiment Ablation Study | Embed ablation bar charts (`accuracy/F1/AUC` & `EER/latency` from `performance_dashboard.py`): Spatial `59.05%` vs Frequency `80.04%` (`0.9092` AUC) vs Exp 5 Full+Reg (`0.822` F1, `19.55%` EER, `10.11 ms`). |
| **8** | Confidence Calibration & 3-Outcome Policy | Embed `reliability_diagram.png` & `pr_threshold_analysis.png`: Validation-only `T = 2.6543`, `tau = 0.71`, `[0.7070, 0.7188]` zone + sharpness/size guards; Validation ECE cut `0.2364 -> 0.1142`; Test NLL cut `0.6247 -> 0.5571`. |
| **9** | Frame-Level Video Forensics (`temporal_modeling=False`) | OpenCV sampling (`fps`/`interval`/`uniform`), Green/Red/Yellow/Gray timeline, 10% trimmed mean + sustained quorum (`>= max(2, 30% N)`) preventing single-frame false positives. |
| **10** | Decoupled Explainable AI (XAI) & Fixes | Multi-face Grad-CAM (`+z`/`-z`), Azimuthal radial spectrum comparison, Q=95/90/75 ELA maps; explicit non-proof SWGDE caveat (heuristic override removed). |
| **11** | Software Forensic Audit Ledger & PDF Certificates | SQLite v2 (`27` cols + `audit_events` `BEFORE UPDATE/DELETE` abort triggers), raw upload SHA-256 vs crop SHA-256, multi-page ReportLab certificates. |
| **12** | Runtime Performance & 73/73 Automated QA | Warmed MPS `batch_size=32` = `219.9 img/s` (`26.4x` vs CPU `8.3 img/s`), MPS->CPU fallback, `73/73` pytest checks across 14 modules including live Streamlit `AppTest`. |
| **13** | Honest Limitations & Scope Boundaries | Unseen-generator false negatives (`morph`/`faceswap`), external FF++/Celeb-DF/DFDC `0/3` locally (`zero fabrication`), `temporal_modeling=False`, ELA washout, software log vs legal TSA custody. |
| **14** | Live Demo & Examiner Q&A | Walkthrough of running Streamlit Studio (`http://localhost:8501`) + quick demo probe results (`authentic` `0.299`, `diffusion` `0.722`, `stylegan` `UNCERTAIN 0.711`). |
