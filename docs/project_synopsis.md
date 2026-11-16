# Final Year B.Tech Capstone Synopsis & Specification

**Project Title:** Deepfake Forensic Sentinel: Dual-Domain Image & Frame-Level Video Deepfake Detection with Calibrated Uncertainty and Explainable Forensics  
**Student Name:** Antriksh Mishra  
**Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Domain:** Computer Vision, Digital Media Forensics, Deep Learning  

---

## 1. Introduction & Problem Statement

Rapid proliferation of GANs and Latent Diffusion portrait generators creates synthetic facial imagery that undermines identity verification and digital media trust. Single-stream RGB CNN detectors frequently memorize donor identities or dataset biases rather than physical synthesis traces, leading to high false-positive rates on unseen identities (`59.05%` accuracy, `77.37%` FPR in our Spatial-Only ablation baseline). Moreover, black-box binary outputs lack calibrated uncertainty handling, frame-level video safeguards, and tamper-evident audit logs.

---

## 2. Project Objectives

1. **Dual-Domain Cross-Attention Architecture:** Fuse `EfficientNet-B0` spatial features (`512`-d) with `2D FFT` + 3 fixed `5x5` SRM high-pass filters (`256`-d) via 4-head Multi-Head Cross-Attention (`DualDomainSentinelNet`, `5,882,013` parameters).
2. **Leakage-Free Dataset & Ablation Rigor:** Eliminate exact and identity source leakage via cryptographic SHA-256 checks, 64-bit 2D DCT pHash clustering, and `StratifiedGroupKFold` (`15` disjoint source groups, `0` leaked groups).
3. **Validation-Only Calibration & 3-Outcome Abstention:** Fit scalar temperature scaling (`T = 2.6543`) and Youden's J thresholding (`tau = 0.71`) strictly on validation logits, routing `[0.7070, 0.7188]` and low-sharpness/tiny crops to `UNCERTAIN / NEEDS MANUAL REVIEW` without heuristic overrides.
4. **Frame-Level Video Forensics & Multi-Face XAI:** Support MP4/AVI/MOV sampling, 10% trimmed mean + sustained quorum aggregation (`temporal_modeling=False`), and per-face Grad-CAM, 2D FFT comparisons, and Q=95/90/75 ELA maps.
5. **Interactive 6-Section Studio, Audit Ledger & PDF Certificates:** Deploy a Streamlit studio backed by SQLite schema v2 (`audit_events` immutability triggers, raw file SHA-256 verification) and multi-page ReportLab certificates.

---

## 3. Verified Deliverables Summary

| Deliverable | Verified Implementation & Empirical Evidence | Status |
| :--- | :--- | :--- |
| **Neural Model & Checkpoint** | `checkpoints/best_model.pth` (`5,882,013` params; Test `75.50%` acc / `0.7396` ROC-AUC @ `tau=0.50`; Calibrated `T=2.6543`, Val `77.25%` acc @ `tau=0.71`) | Verified |
| **Dataset Audit & Ablation** | `2,400` images, `0` exact duplicates, `0` leaked groups; 5-experiment ablation (`Exp 5: 79.84% Acc, 0.822 F1, 19.55% EER`) | Verified |
| **Calibration & Video Engine** | Validation-only Temperature Scaling & 3-outcome policy + OpenCV frame-level analyzer with single-frame spike immunity | Verified |
| **Interactive 6-Section UI & PDFs** | Streamlit Studio (`app/app.py`) + multi-page Image & Video ReportLab certificates + SQLite `audit_events` triggers | Verified |
| **Automated QA Suite** | **73 / 73** pytest tests passing across `14` modules (including batched MPS/CPU parity & live Streamlit `AppTest` E2E) | Passing |
| **Capstone Defense Kit** | [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md), [`METHODOLOGY.md`](METHODOLOGY.md), [`DATASET_AND_EVALUATION.md`](DATASET_AND_EVALUATION.md), [`TESTING_REPORT.md`](TESTING_REPORT.md), [`USER_GUIDE.md`](USER_GUIDE.md), [`TECHNICAL_LIMITATIONS.md`](TECHNICAL_LIMITATIONS.md), [`CAPSTONE_SUBMISSION_KIT.md`](CAPSTONE_SUBMISSION_KIT.md), [`viva_questions.md`](viva_questions.md) | Complete |

**Submitted by:** Antriksh Mishra — Final Year B.Tech, Computer Science & Engineering
