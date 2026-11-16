# Operator & Developer User Guide — Deepfake Forensic Sentinel

**Author:** Antriksh Mishra (Final Year B.Tech CSE Capstone)  
**Target Interface:** Unified 6-Section Streamlit Forensic Studio (`http://localhost:8501`) + Python CLI Modules

---

## 1. Prerequisites & Environment Setup

1. **System Requirements:** Python 3.10+ (verified on Python 3.12.6, macOS arm64 Apple Silicon MPS / CUDA / CPU), minimum 8 GB RAM (16 GB recommended).
2. **Create and Activate Virtual Environment:**
   ```bash
   cd /Users/antrikshmishra/Desktop/dfake
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Optional External Dataset Placement (Zero Training, Read-Only Evaluation):**
   If you have official access to public deepfake benchmarks, place them under `data/external/` (or export `FFPP_DIR`, `CELEBDF_DIR`, `DFDC_DIR`):
   - `data/external/FaceForensics++/` (`original_sequences/` & `manipulated_sequences/` or `real/` & `fake/`)
   - `data/external/Celeb-DF/` (`Celeb-real/`, `YouTube-real/`, `Celeb-synthesis/`)
   - `data/external/DFDC/` (clips with `metadata.json` / `labels.csv`)

---

## 2. Launching the Streamlit Forensic Studio

Start the headless Streamlit server with explicit cache/home environment variables for reliable Matplotlib and PyTorch behavior:

```bash
TORCH_HOME=/Users/antrikshmishra/.cache/torch \
HOME=/Users/antrikshmishra/Desktop/dfake \
MPLCONFIGDIR=/Users/antrikshmishra/Desktop/dfake/.matplotlib \
.venv/bin/streamlit run app/app.py --server.port 8501 --server.headless true
```

Open **[http://localhost:8501](http://localhost:8501)** in any modern browser.

> **Test Isolation Tip:** Developers can set `DFAKE_DB_PATH` and `DFAKE_REPORT_PATH` environment variables prior to launch to isolate SQLite writes and generated PDFs away from `data/forensic_history.db`.

---

## 3. Walkthrough of the 6 Workspace Sections

Use the horizontal navigation bar at the top of the studio to switch between sections. The persistent **left sidebar** displays active hardware (`MPS` / `CUDA` / `CPU`), live CPU process RSS and accelerator RAM, calibrated operating constants (`T=2.65`, `tau=0.71`), and active SQLite case counts.

### Section 1: Overview (`🏠 Overview`)
- Displays four top-level status cards: **Production Checkpoint** (`checkpoints/best_model.pth`, SHA-256 prefix), **Validated Calibration** (`T=2.6543`, `tau=0.71`, Band `[0.707, 0.719]`), **Source-Disjoint Dataset Audit** (`2,400` images, `15` groups, `0` leaked), and **Forensic Audit Ledger** active counts.
- Provides quick-start jump buttons directly into Image Analysis, Video Analysis, Model Performance, and Forensic Evidence.

### Section 2: Image Analysis (`🖼️ Image Analysis`)
1. **Select Input Mode:** Choose **Upload Image** (`JPG`, `PNG`, `JPEG`, `WEBP`), **Live Webcam Capture**, or **Preloaded Benchmark Gallery** (`data/demo_samples/*.jpg`).
2. **Intake Logging:** Immediately upon upload/selection, the system hashes the **raw uploaded bytes** (`EVD-...`), records dimensions and byte size, and inserts an immutable `upload` event into `audit_events`.
3. **Run Examination:** Click **`🚀 Run Forensic Examination`**. The studio outputs:
   - Color-coded **3-Outcome Verdict Banner**: Green (`REAL` / `AUTHENTIC`), Amber (`UNCERTAIN`), or Red (`FAKE` / `MANIPULATED`), accompanied by Case ID, Evidence ID, raw SHA-256, calibration band, latency breakdown, and batched hardware telemetry.
   - **Intelligent Media Localization**: Automatic detection of human facial landmarks or salient non-face object boundaries (for sneakers, products, animals, vehicles, and artwork).
   - **Decoupled Auxiliary Signals**: Calibrated neural classifier progress bar alongside independent advisory 2D FFT spectral, Spatial SRM noise residuals, and JPEG/PNG ELA indicators (`heuristic_override_applied = False`).
   - **Unified Forensic Evidence Panel**: Interactive per-region selector displaying localized media boxes, Spatial Grad-CAM (face zones or structural object zones), Spatial SRM Noise Residuals ($F_1, F_2, F_3$), 2D FFT Inferno map, Azimuthal Radial Power Comparison with adaptive optical decay, format-aware ELA Color Map, and structured forensic observations.
4. **Export Certificate:** Click **`📑 Compile Forensic PDF Certificate`**, then **`⬇️ Download Forensic Examination Certificate (PDF)`** (`report_generation` event is automatically appended to `audit_events`).

### Section 3: Video Analysis (`🎥 Video Analysis`)
1. **Upload Video:** Upload an `.mp4`, `.avi`, or `.mov` clip (up to `250 MB`). Validation rejects empty, corrupted, or unsupported containers cleanly before logging an intake `upload` event.
2. **Configure Frame Sampling:** Choose **Sampling Mode** (`fps`, `interval`, or `uniform`), sampling rate, and **Max Sampled Frames**.
3. **Analyze & Inspect:** Click **`🚀 Run Frame-Level Video Forensic Analysis`**:
   - Displays the explicit **`temporal_modeling = False`** methodological disclosure banner.
   - Summarizes trimmed-mean score, sustained multi-frame quorum status (`>= max(2, ceil(0.30 * N))`), and isolated single-frame spike suppression.
   - **Color-Coded Timeline & Confidence Distribution:** Interactive Matplotlib timeline highlighting `Authentic (Green)`, `Manipulated (Red)`, `Uncertain (Yellow)`, and `No Face Detected (Gray)` across timestamps.
   - **Suspicious Intervals & Clickable Frame XAI:** Select any timestamp frame to view bounding boxes and optionally click **`🔬 Run On-Demand Frame XAI Inspection`** to launch Grad-CAM/FFT/ELA for that frame in Section 5.
   - **Download Exports:** Multi-page Video **PDF Certificate**, full **JSON report**, or structured **Markdown summary**.

### Section 4: Model Performance (`📊 Model Performance`)
- Automatically refreshes via `st_mtime_ns` + `st_size` signature watching (**B10 fix**).
- Displays **Held-Out Test (`N=400`) vs Validation (`N=400`)** metric cards, PNG Confusion Matrices and ROC curves, **Training vs Validation Learning Curves**, **Validation Calibration Diagrams** (Reliability Diagram, Confidence Distribution with uncertainty band, PR Threshold Curve), **5-Experiment Ablation Study** charts and table, **Dataset Leakage Audit**, and **External Benchmark Status** (`0/3` available locally).

### Section 5: Forensic Evidence (`🔬 Forensic Evidence`)
Organized into two deep-dive tabs:
1. **Active Examination XAI Workbench:** Deep-dive tabs for **Spatial Grad-CAM Details** (tabular ocular/mid-facial/jawline activation pooling across multi-face crops), **2D FFT Radial Spectrum Comparison** (query curve vs authentic reference band, optical `1/f` decay, anomalous frequency rings), **Multi-Quality ELA & Limitations** (side-by-side `Q=95`, `Q=90`, `Q=75` maps + explicit JPEG washout caveats), and PDF export.
2. **SQLite Forensic Audit Ledger & Raw File Hash Verifier:**
   - Displays software audit log notice (`CHAIN_OF_CUSTODY_DISCLAIMER`).
   - Browse active/archived examinations (filter by `image` / `video`), inspect per-case `audit_events` lifecycle timeline (`upload -> processing -> prediction -> report_generation -> archive_action`).
   - **Raw File Integrity Check:** Re-upload any candidate media file inside **`🔐 Verify Raw Uploaded File Integrity Against Stored SHA-256`** to detect even a 1-byte modification (`MATCH` vs `MISMATCH_TAMPERED`).
   - **CSV Export & Two-Step Soft Archiving:** Check confirmation box -> click **`🗄️ Archive Active History`** (marks `is_archived = 1`; never deletes immutable `audit_events`).

### Section 6: Model Information (`ℹ️ Model Information`)
- Inspects live PyTorch submodule parameter counts (`5,882,013` total), full checkpoint SHA-256 digest, calibration policy constants, active hardware memory telemetry (`cpu_rss_mb`, `accelerator_alloc_mb`), and full **Known Scientific Limitations**.

---

## 4. Command-Line Workflows (Scripts & Tests)

| Task | Command |
| :--- | :--- |
| **Read-Only Dataset Leakage Audit** | `.venv/bin/python -m src.data.dataset_audit --data_dir data` |
| **Batched Checkpoint Evaluation** | `.venv/bin/python -m src.evaluation.evaluate_checkpoint --checkpoint checkpoints/best_model.pth` |
| **Validation Calibration & Policy Fit** | `.venv/bin/python -m src.evaluation.calibration --checkpoint checkpoints/best_model.pth` |
| **5-Experiment Ablation Runner** | `.venv/bin/python -m src.evaluation.ablation_study` |
| **External Benchmarks Evaluator** | `.venv/bin/python -m src.evaluation.external_benchmark --checkpoint checkpoints/best_model.pth` |
| **Complete 73-Test QA Suite** | `TORCH_HOME=~/.cache/torch HOME=/Users/antrikshmishra/Desktop/dfake MPLCONFIGDIR=/Users/antrikshmishra/Desktop/dfake/.matplotlib .venv/bin/pytest tests/ -q` |
