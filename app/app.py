"""
Deepfake Forensic Sentinel - Unified 6-Section Streamlit Studio.

Sections:
  1. Overview          - Executive workflow, live calibration readiness, dataset audit snapshot
  2. Image Analysis    - Upload / webcam / benchmark image examination, 3-outcome policy, Unified 8-Item Panel & PDF
  3. Video Analysis    - Frame-level MP4/AVI/MOV forensic timeline, click-to-inspect frames & PDF/JSON/MD export
  4. Model Performance - Held-out test metrics, train/val curves, calibration/PR diagrams, 5-exp ablation & external status
  5. Forensic Evidence - Deep multi-face XAI workbench (Grad-CAM, FFT radial, ELA Q95/90/75) + SQLite ledger & hash verifier
  6. Model Information - Dynamic parameter counts, checkpoint provenance, hardware runtime & scientific limitations

Lead Developer: Antriksh Mishra (Final Year B.Tech CSE Capstone)
"""

import io
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project repository root is always in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hardened runtime environment for instant startup on macOS:
# 1. Direct MPLCONFIGDIR to pre-compiled local font cache (.matplotlib) to avoid 70s+ font scan hang
mpl_cache = REPO_ROOT / ".matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
import pandas as pd
from PIL import Image
import streamlit as st
import torch

try:
    from app.performance_dashboard import load_evaluation_data, render_performance_dashboard
except (ImportError, ModuleNotFoundError):
    from performance_dashboard import load_evaluation_data, render_performance_dashboard

try:
    from app.report_generator import generate_pdf_report
except (ImportError, ModuleNotFoundError):
    from report_generator import generate_pdf_report

try:
    from app.video_dashboard import render_video_analysis_view
except (ImportError, ModuleNotFoundError):
    from video_dashboard import render_video_analysis_view

try:
    from app.xai_panel import (
        get_active_face_report,
        render_ela_tab,
        render_spatial_gradcam_tab,
        render_spectral_fft_tab,
        render_srm_tab,
        render_unified_forensic_panel,
    )
except (ImportError, ModuleNotFoundError):
    from xai_panel import (
        get_active_face_report,
        render_ela_tab,
        render_spatial_gradcam_tab,
        render_spectral_fft_tab,
        render_srm_tab,
        render_unified_forensic_panel,
    )

from src.core.database import (
    CHAIN_OF_CUSTODY_DISCLAIMER,
    ForensicDatabase,
    compute_bytes_sha256,
)
from src.core.utils import get_device, get_runtime_memory_telemetry
from src.pipeline import ForensicInferencePipeline


SECTION_OVERVIEW = "🏠 Overview"
SECTION_IMAGE = "🖼️ Image Analysis"
SECTION_VIDEO = "🎥 Video Analysis"
SECTION_PERFORMANCE = "📊 Model Performance"
SECTION_EVIDENCE = "🔬 Forensic Evidence"
SECTION_INFO = "ℹ️ Model Information"

NAV_SECTIONS: List[str] = [
    SECTION_OVERVIEW,
    SECTION_IMAGE,
    SECTION_VIDEO,
    SECTION_PERFORMANCE,
    SECTION_EVIDENCE,
    SECTION_INFO,
]


st.set_page_config(
    page_title="Deepfake Forensic Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.15rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.0rem;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.28);
    }
    .verdict-real {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.28);
    }
    .verdict-inconclusive {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.28);
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .workflow-card {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-top: 4px solid #2563eb;
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 155px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> ForensicInferencePipeline:
    ckpt = REPO_ROOT / "checkpoints" / "best_model.pth"
    checkpoint_path = str(ckpt) if ckpt.exists() else None
    return ForensicInferencePipeline(checkpoint_path=checkpoint_path)


@st.cache_resource
def load_database() -> ForensicDatabase:
    return ForensicDatabase()


pipeline = load_pipeline()
device = get_device()
db = load_database()

# Initialize persistent workspace state
defaults = {
    "nav_section": SECTION_OVERVIEW,
    "analysis_results": None,
    "analyzed_image_source": None,
    "analyzed_raw_sha256": None,
    "pdf_ready_path": None,
    "ev_pdf_ready_path": None,
    "active_img_upload_sig": None,
    "active_evidence_id": None,
    "active_raw_sha256": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _jump_to_section(section_name: str) -> None:
    st.session_state["nav_section"] = section_name


def _fetch_case_events(case_id: Optional[str], evidence_id: Optional[str]) -> List[Dict[str, Any]]:
    if evidence_id and not str(evidence_id).startswith("EVD-LEGACY-"):
        raw_ev = db.get_audit_events(evidence_id=evidence_id)
        return [ev for ev in raw_ev if ev.get("case_id") in (case_id, None)]
    if case_id:
        return db.get_audit_events(case_id=case_id)
    return []


def _get_pdf_report_path() -> Path:
    env_path = os.environ.get("DFAKE_REPORT_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return REPO_ROOT / "forensic_report.pdf"


def _compile_and_log_image_pdf(results: Dict[str, Any], out_name: str) -> bytes:
    events = _fetch_case_events(results.get("case_id"), results.get("evidence_id"))
    pdf_path = _get_pdf_report_path()
    generate_pdf_report(results, output_path=str(pdf_path), audit_events=events)
    pdf_bytes = pdf_path.read_bytes()
    db.record_report_generation(
        case_id=results.get("case_id"),
        evidence_id=results.get("evidence_id"),
        report_format="pdf",
        report_bytes=pdf_bytes,
        output_name=out_name,
        actor="streamlit_pdf_exporter",
    )
    return pdf_bytes


# Sidebar
with st.sidebar:
    st.markdown(
        '<div style="font-size:2.4rem;line-height:1.1;margin-bottom:4px;">🛡️</div>',
        unsafe_allow_html=True,
    )
    st.title("Forensic Sentinel")
    st.caption("Dual-Domain Computer Vision Capstone")

    st.markdown("---")
    st.markdown("### 👨‍💻 Lead Developer")
    st.markdown("**Antriksh Mishra**")
    st.caption("Final Year B.Tech • Computer Science & Engineering\nSpecialization: Computer Vision & Deep Learning")

    st.markdown("---")
    st.markdown("### 🖥️ Calibrated Runtime")
    cal_meta_sb = getattr(pipeline, "calibration_info", {}) or {}
    mem_sb = get_runtime_memory_telemetry(getattr(pipeline, "device", device))
    accel_line = (
        f"**{mem_sb['device'].upper()} Alloc:** `{mem_sb['accelerator_alloc_mb']:.1f} MB`\n\n"
        if mem_sb["device"] != "cpu"
        else ""
    )
    st.info(
        f"**Device:** `{mem_sb['device'].upper()}`\n\n"
        f"**Process RAM:** `{mem_sb['cpu_rss_mb']:.1f} MB`\n\n"
        f"{accel_line}"
        f"**Backbone:** `EfficientNet-B0 + 3 SRM`\n\n"
        f"**Fusion:** `Cross-Attention (4-Head)`\n\n"
        f"**Calibration:** `T={float(getattr(pipeline, 'temperature', 1.0)):.2f}`, "
        f"`tau={float(cal_meta_sb.get('decision_threshold', 0.50)):.2f}`"
    )

    stats_sb = db.get_summary_stats(include_archived=False)
    st.markdown("### 📜 Active Ledger")
    st.caption(
        f"**{stats_sb['total']}** active case(s) "
        f"(`{stats_sb['images']}` img / `{stats_sb['videos']}` vid)"
    )

    st.markdown("---")
    st.markdown("### 🎓 Scientific Safeguards")
    st.markdown(
        "- **Decoupled Classifier**: Zero heuristic override on neural logits.\n"
        "- **3-Outcome Policy**: Authentic, Manipulated, or Manual Review.\n"
        "- **Frame-Level Video**: Explicit `temporal_modeling=False` scope.\n"
        "- **Immutable Events**: Append-only `audit_events` SQLite ledger."
    )
    st.markdown("---")
    st.caption("Developed by Antriksh Mishra © 2026")


# Header & Global 6-Section Navigation Bar
st.markdown('<div class="main-header">🛡️ Deepfake Forensic Sentinel</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Dual-Domain Spatial &amp; 2D FFT Spectral Deepfake Detection with Explainable AI Forensics &nbsp;|&nbsp; '
    '<b>Lead Developer: Antriksh Mishra</b>'
    '</div>',
    unsafe_allow_html=True,
)

active_section = st.radio(
    "Workspace Navigation:",
    NAV_SECTIONS,
    horizontal=True,
    key="nav_section",
    label_visibility="collapsed",
)
st.markdown("---")


# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
def render_overview_section() -> None:
    st.subheader("🏠 System Overview & Forensic Architecture")
    st.write(
        "**Deepfake Forensic Sentinel** is an end-to-end multimodal media integrity workbench combining "
        "spatial RGB representation learning (**EfficientNet-B0**) with frequency-domain noise-residual extraction "
        "(**2D FFT Log-Magnitude + 3 Fixed SRM High-Pass Filters**) via **Multi-Head Cross-Attention Fusion**. "
        "Predictions are temperature-scaled strictly on held-out validation logits with no test-set tuning."
    )

    eval_data = load_evaluation_data()
    ckpt = eval_data.get("checkpoint", {})
    cal = eval_data.get("calibration", {})
    audit = eval_data.get("dataset_audit", {})
    stats = db.get_summary_stats(include_archived=False)
    pol = cal.get("uncertainty_policy", {})
    temp_val = cal.get("temperature_scaling", {}).get("temperature", getattr(pipeline, "temperature", 1.0))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Production Checkpoint",
        f"{ckpt.get('size_mb', 0.0)} MB" if ckpt.get("exists") else "Missing",
        delta=f"SHA: {ckpt.get('sha256', 'N/A')[:12]}...",
        delta_color="off",
    )
    k2.metric(
        "Validated Calibration",
        f"tau = {float(pol.get('decision_threshold', 0.50)):.2f}",
        delta=f"T = {float(temp_val):.4f} | Band=[{float(pol.get('uncertain_low', 0.40)):.3f}, {float(pol.get('uncertain_high', 0.60)):.3f}]",
        delta_color="off",
    )
    src_groups = audit.get("source_group_analysis", {}).get("total_source_groups", "N/A")
    leaked_groups = audit.get("source_group_analysis", {}).get("cross_split_leaked_group_count", "N/A")
    k3.metric(
        "Source-Disjoint Dataset Audit",
        f"{audit.get('total_image_count', 0):,} Images" if audit else "Not Loaded",
        delta=f"{src_groups} source groups | {leaked_groups} leaked groups",
        delta_color="off",
    )
    k4.metric(
        "Forensic Audit Ledger",
        f"{stats['total']} Active Case(s)",
        delta=f"{stats['images']} Image / {stats['videos']} Video",
        delta_color="off",
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="workflow-card">'
            '<h4>1️⃣ Intake &amp; Cryptographic Hashing</h4>'
            '<p style="font-size:0.9rem;color:#334155;">Computes raw uploaded bytes SHA-256 prior to preprocessing, '
            'records dimensions and size, extracts Haar faces with eye + skin-ratio validation (B4 fixed), '
            'and preserves aligned crops separately.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="workflow-card">'
            '<h4>2️⃣ Dual-Domain Cross-Attention</h4>'
            '<p style="font-size:0.9rem;color:#334155;">Couples frozen/fine-tuned EfficientNet-B0 spatial descriptors '
            'with SRM-filtered 2D FFT spectral embeddings. Outputs are temperature-scaled into the 3-outcome '
            'policy with zero heuristic override.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="workflow-card">'
            '<h4>3️⃣ XAI, Video &amp; Audit Certificates</h4>'
            '<p style="font-size:0.9rem;color:#334155;">Generates advisory per-face Grad-CAM, 2D FFT radial power comparisons, '
            'multi-quality ELA maps, interactive Green/Red/Yellow video timelines, immutable SQLite '
            '<code>audit_events</code>, and multi-page PDF certificates.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 🚀 Quick Start Modules")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.button("🖼️ Open Image Analysis", on_click=_jump_to_section, args=(SECTION_IMAGE,), width="stretch", type="primary")
    with q2:
        st.button("🎥 Open Video Analysis", on_click=_jump_to_section, args=(SECTION_VIDEO,), width="stretch")
    with q3:
        st.button("📊 Open Model Performance", on_click=_jump_to_section, args=(SECTION_PERFORMANCE,), width="stretch")
    with q4:
        st.button("🔬 Open Forensic Evidence", on_click=_jump_to_section, args=(SECTION_EVIDENCE,), width="stretch")

    st.markdown("---")
    st.markdown("### ⚖️ Scientific Scope & Forensic Guarantees")
    st.markdown(
        """
| Module Area | Verified Operational Scope |
| :--- | :--- |
| **Classifier Policy** | Temperature scaling (`T=2.6543`) and threshold (`tau=0.71`) fit strictly on validation logits; outputs `AUTHENTIC`, `MANIPULATED`, or `UNCERTAIN / NEEDS MANUAL REVIEW`. |
| **Explainable AI (XAI)** | Grad-CAM, 2D FFT radial profiles, and ELA Q95/90/75 heatmaps act strictly as interpretive diagnostic corroboration — never physical proof of forgery. |
| **Video Forensics** | Frame-level dual-domain inference (`analysis_scope="frame_level_inference"`, `temporal_modeling=False`) with single-frame spike suppression quorum. |
| **Audit Integrity** | Local software forensic audit logs with append-only `audit_events` triggers; explicitly distinct from legally admissible hardware/TSA chain of custody. |
"""
    )


# ==============================================================================
# SECTION 2: IMAGE ANALYSIS
# ==============================================================================
def render_image_analysis_section() -> None:
    st.subheader("🖼️ Image Forensic Examination")
    input_mode = st.radio(
        "Choose Image Input Mode:",
        ["📁 Upload Image", "📸 Live Webcam Capture", "🧪 Preloaded Benchmark Gallery"],
        horizontal=True,
        key="img_input_mode_selector",
    )

    input_image: Optional[Image.Image] = None
    image_source_name = "Image"
    raw_file_bytes: Optional[bytes] = None
    file_size_bytes = 0

    if input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Face Image (JPG, PNG, WEBP)",
            type=["jpg", "png", "jpeg", "webp"],
            key="img_file_uploader",
        )
        if uploaded_file is not None:
            raw_file_bytes = uploaded_file.getvalue()
            file_size_bytes = len(raw_file_bytes)
            input_image = Image.open(io.BytesIO(raw_file_bytes)).convert("RGB")
            image_source_name = uploaded_file.name
        else:
            st.info("👆 Upload a face portrait above or switch to the Preloaded Benchmark Gallery to begin.")

    elif input_mode == "📸 Live Webcam Capture":
        camera_file = st.camera_input("Capture Live Portrait from Webcam", key="img_camera_input")
        if camera_file is not None:
            raw_file_bytes = camera_file.getvalue()
            file_size_bytes = len(raw_file_bytes)
            input_image = Image.open(io.BytesIO(raw_file_bytes)).convert("RGB")
            image_source_name = getattr(camera_file, "name", None) or "Live Webcam Portrait"
        else:
            st.info("👆 Allow camera access and capture a portrait frame.")

    elif input_mode == "🧪 Preloaded Benchmark Gallery":
        demo_dir = REPO_ROOT / "data" / "demo_samples"
        if demo_dir.exists():
            demo_files = sorted(list(demo_dir.glob("*.jpg")))
            if demo_files:
                selected_sample = st.selectbox(
                    "Select a benchmark sample to test:",
                    [f.name for f in demo_files],
                    key="sample_selector",
                )
                if selected_sample:
                    sample_path = demo_dir / selected_sample
                    raw_file_bytes = sample_path.read_bytes()
                    file_size_bytes = len(raw_file_bytes)
                    input_image = Image.open(io.BytesIO(raw_file_bytes)).convert("RGB")
                    image_source_name = selected_sample
            else:
                st.warning("No samples found in `data/demo_samples`.")
        else:
            st.warning("Demo samples directory not initialized.")

    if input_image is None:
        return

    media_dimensions = f"{input_image.width}x{input_image.height}"
    current_raw_sha = compute_bytes_sha256(raw_file_bytes) if raw_file_bytes is not None else "N/A"
    upload_sig = f"{input_mode}:{image_source_name}:{current_raw_sha}"

    if st.session_state.get("active_img_upload_sig") != upload_sig:
        intake = db.record_media_upload(
            filename=image_source_name,
            raw_file_bytes=raw_file_bytes,
            raw_file_sha256=current_raw_sha,
            file_size_bytes=file_size_bytes,
            media_type="image",
            media_dimensions=media_dimensions,
            actor="streamlit_image_ui",
        )
        st.session_state["active_img_upload_sig"] = upload_sig
        st.session_state["active_evidence_id"] = intake["evidence_id"]
        st.session_state["active_raw_sha256"] = intake["raw_file_sha256"]

    st.markdown("---")
    col_input, col_action = st.columns([1, 2])
    with col_input:
        st.image(input_image, caption=f"Input Media: {image_source_name} ({media_dimensions})", width="stretch")
    with col_action:
        st.write("### Ready for Inspection")
        st.write(
            "Runs `DualDomainSentinelNet` calibrated inference, multi-face localization, "
            "Grad-CAM attribution, 2D FFT azimuthal radial profiling, and multi-quality ELA residuals."
        )
        st.caption(
            f"Evidence ID: `{st.session_state.get('active_evidence_id', 'Pending')}` &bull; "
            f"Raw SHA-256: `{current_raw_sha[:16]}...` (`{file_size_bytes:,}` bytes)"
        )
        op_mode_choice = st.selectbox(
            "Operating Decision Mode:",
            [
                "Balanced (τ=0.71, Validated)",
                "High-Recall / Forensic Audit (τ=0.34, Catches subtle fakes)",
                "High-Precision (τ=0.72, FPR ≤ 3%)",
            ],
            index=0,
            key="img_operating_mode_selector",
            help="Balanced: optimal Youden-J threshold with cross-domain conflict protection. High-Recall: prioritizes detecting AI generation by overriding suppressed spatial logits. High-Precision: strictly protects against false positives.",
        )
        mode_key = "balanced"
        if "High-Recall" in op_mode_choice:
            mode_key = "high_recall"
        elif "High-Precision" in op_mode_choice:
            mode_key = "high_precision"

        run_btn = st.button("🚀 Run Forensic Examination", type="primary", width="stretch", key="run_img_exam_btn")

    if run_btn:
        with st.spinner("Executing Dual-Domain Forensic Analysis..."):
            res = pipeline.analyze_image(input_image, operating_mode=mode_key)
            case_id = db.log_examination(
                res,
                image_source_name=image_source_name,
                device_str=device.type.upper(),
                raw_file_bytes=raw_file_bytes,
                file_size_bytes=file_size_bytes,
                media_type="image",
                media_dimensions=media_dimensions,
                evidence_id=st.session_state.get("active_evidence_id"),
                upload_already_logged=bool(st.session_state.get("active_evidence_id")),
                actor="streamlit_image_ui",
            )
            res["case_id"] = case_id
            st.session_state["analysis_results"] = res
            st.session_state["analyzed_image_source"] = image_source_name
            st.session_state["analyzed_raw_sha256"] = current_raw_sha
            st.session_state["pdf_ready_path"] = None

    if (
        st.session_state.get("analysis_results") is not None
        and st.session_state.get("analyzed_image_source") == image_source_name
        and st.session_state.get("analyzed_raw_sha256") == current_raw_sha
    ):
        results = st.session_state["analysis_results"]

        verdict_badge = results.get("verdict_badge", "FORENSIC VERDICT")
        verdict_class = results.get("verdict_class", "verdict-inconclusive")
        verdict_exp = results.get("verdict_explanation", "Dual-domain calibrated examination completed.")
        conf = float(results["confidence_percent"])
        ref_id = results.get("case_id", "DFS-LIVE")
        evd_id = results.get("evidence_id", "EVD-LIVE")
        raw_sha_disp = results.get("raw_file_sha256", "N/A")
        cal_meta = results.get("calibration_metadata", {})
        cal_note = (
            f"T={cal_meta.get('temperature', 1.0):.2f} | tau={cal_meta.get('decision_threshold', 0.50):.2f} "
            f"| Band=[{cal_meta.get('uncertain_low', 0.40):.3f}, {cal_meta.get('uncertain_high', 0.60):.3f}]"
            if cal_meta else "Uncalibrated T=1.00"
        )

        if results.get("false_negative_risk") and results.get("false_negative_warning"):
            st.warning(f"⚠️ **False-Negative Risk Alert:** {results['false_negative_warning']}")

        st.markdown(
            f'<div class="{verdict_class}">{verdict_badge}<br/>'
            f'<span style="font-size: 1.02rem; font-weight: 400;">Classifier Confidence: <b>{conf:.2f}%</b> &nbsp;|&nbsp; Calibrated P(Manipulated): <b>{results["fake_probability"]:.4f}</b> &nbsp;|&nbsp; Case: <code>{ref_id}</code> &nbsp;|&nbsp; Evidence: <code>{evd_id}</code></span>'
            f'<div style="font-size: 0.78rem; font-weight: 400; margin-top: 4px; opacity: 0.9;">Raw File SHA-256: <code>{raw_sha_disp}</code> | Calibration Policy: {cal_note}</div>'
            f'<div style="font-size: 0.88rem; font-weight: 400; margin-top: 6px; opacity: 0.92;">{verdict_exp}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="disclaimer-box">'
            '<b>⚖️ Forensic Science Principle:</b> Per SWGDE / ENFSI standards, calibrated neural scores provide probabilistic corroboration. '
            'Auxiliary 2D FFT and ELA indicators are displayed strictly independently without heuristic override.'
            '</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calibrated P(Authentic)", f"{results['real_probability'] * 100:.1f}%")
        m2.metric("Calibrated P(Manipulated)", f"{results['fake_probability'] * 100:.1f}%")
        m3.metric("ELA Mean Error", f"{results['ela_mean_error']:.2f}")
        m4.metric("Inference Latency", f"{results['inference_time_ms']:.1f} ms")

        # Dual-Domain Stream Contribution Breakdown
        s_fake = results.get("spatial_fake_prob")
        f_fake = results.get("frequency_fake_prob")
        disc_val = results.get("branch_discrepancy", 0.0)
        has_conf = results.get("has_branch_conflict", False)

        if s_fake is not None and f_fake is not None:
            st.markdown("##### 🔬 Dual-Domain Stream Contribution Breakdown")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Spatial Stream (RGB)", f"{s_fake * 100:.1f}% Fake", help="EfficientNet-B0 feature stream")
            sc2.metric("Frequency Stream (FFT+SRM)", f"{f_fake * 100:.1f}% Fake", help="SRM residuals + 2D FFT spectral stream")
            sc3.metric(
                "Domain Divergence",
                f"{disc_val * 100:.1f}%",
                delta="Elevated Divergence" if disc_val >= 0.20 else "Corroborated",
                delta_color="inverse" if disc_val >= 0.20 else "normal",
            )
            sc4.metric("Stream Consistency", "⚠️ CONFLICT DETECTED" if has_conf else "✅ CORROBORATED")

        lat = results.get("latency_breakdown", {})
        tel = results.get("performance_telemetry", {})
        if lat:
            st.caption(
                f"⏱️ **Hardware Latency Breakdown:** Preprocessing: `{lat.get('preprocessing_ms', lat.get('face_detection_ms', 0))}ms` &bull; "
                f"Neural Inference: `{lat.get('neural_inference_ms', 0)}ms` &bull; "
                f"Explainability (FFT+ELA+SRM+CAM): `{lat.get('explainability_ms', 0)}ms` &bull; "
                f"Postprocessing: `{lat.get('postprocessing_ms', 0)}ms` &bull; "
                f"Total End-to-End: `{lat.get('total_ms', 0)}ms`"
            )
        if tel:
            fb_note = " | CPU fallback triggered" if tel.get("cpu_fallback_triggered") else ""
            st.caption(
                f"⚡ **Batch & Memory Telemetry:** Device `{str(tel.get('device', 'cpu')).upper()}` &bull; "
                f"Batch Size: `{tel.get('batch_size', 1)}` &bull; "
                f"Pipeline Throughput: `{tel.get('throughput_fps', 0.0):.2f} images/s` &bull; "
                f"Neural Throughput: `{tel.get('neural_throughput_fps', 0.0):.2f} img/s` &bull; "
                f"CPU RAM: `{tel.get('cpu_rss_mb', 0.0):.1f} MB` &bull; "
                f"Accel Alloc: `{tel.get('accelerator_alloc_mb', 0.0):.1f} MB`{fb_note}"
            )

        multi_faces = results.get("multi_face_reports", [])
        if len(multi_faces) > 1:
            with st.expander(f"👥 Multi-Face Breakdown ({len(multi_faces)} Faces Detected)", expanded=True):
                cols_mf = st.columns(min(len(multi_faces), 4))
                for i, f_rep in enumerate(multi_faces):
                    with cols_mf[i % len(cols_mf)]:
                        st.image(
                            f_rep["aligned_face"],
                            caption=f"{'Primary Face' if f_rep['is_primary'] else f'Face #{i+1}'} ({f_rep['bbox'][2]}x{f_rep['bbox'][3]})",
                            width="stretch",
                        )
                        st.markdown(f"**{f_rep['verdict_badge']}**")
                        st.caption(f"Conf: **{f_rep['confidence_percent']:.1f}%** | P(Fake): **{f_rep['fake_probability']:.3f}**")

        ev = results.get("forensic_evidence", {})
        if ev:
            with st.expander("🔍 Classifier Confidence & Decoupled Auxiliary Forensic Evidence", expanded=True):
                e_col1, e_col2, e_col3 = st.columns(3)
                with e_col1:
                    st.markdown("**🧠 Primary Neural Classifier (Calibrated)**")
                    p_n = float(ev.get("neural_fake_prob", 0.5))
                    p_uncal = float(ev.get("neural_uncalibrated_prob", p_n))
                    st.progress(p_n)
                    st.write(f"Calibrated P(Manipulated): **{p_n*100:.1f}%** *(Uncal: {p_uncal*100:.1f}%)*")
                    st.caption(f"Outcome: `{results.get('prediction_outcome', 'N/A')}` | Entropy: `{cal_meta.get('predictive_entropy_bits', 0.0):.3f} bits`")
                with e_col2:
                    st.markdown("**📊 Auxiliary 2D FFT Spectral Evidence**")
                    p_s = float(ev.get("spectral_anomaly_prob", 0.5))
                    st.progress(p_s)
                    st.write(f"Spectral Discrepancy: **{p_s*100:.1f}%**")
                    st.caption("Independent radial FFT high-frequency signal (advisory only).")
                with e_col3:
                    st.markdown("**🔍 Auxiliary Quantization Residual (ELA)**")
                    p_e = float(ev.get("ela_anomaly_prob", 0.5))
                    st.progress(p_e)
                    st.write(f"Compression Mismatch: **{p_e*100:.1f}%**")
                    st.caption("Independent Q=90 JPEG spatial residual variance (advisory only).")

        active_rep = get_active_face_report(results, key_prefix="img_sec")
        render_unified_forensic_panel(results, active_rep)

        st.markdown("---")
        act_c1, act_c2 = st.columns([1.2, 1.0])
        with act_c1:
            if st.button("📑 Compile Forensic PDF Certificate", key="gen_pdf", width="stretch", type="primary"):
                with st.spinner("Generating multi-page PDF Forensic Certificate..."):
                    pdf_bytes = _compile_and_log_image_pdf(
                        results, f"Deepfake_Forensic_Report_{image_source_name}.pdf"
                    )
                    st.session_state["pdf_ready_path"] = str(_get_pdf_report_path())
                    st.success(f"Certificate generated ({len(pdf_bytes):,} bytes) and recorded in `audit_events`.")
            if st.session_state.get("pdf_ready_path") and Path(st.session_state["pdf_ready_path"]).exists():
                st.download_button(
                    label="⬇️ Download Forensic Examination Certificate (PDF)",
                    data=Path(st.session_state["pdf_ready_path"]).read_bytes(),
                    file_name=f"Deepfake_Forensic_Report_{image_source_name}.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="dl_img_pdf_main",
                )
        with act_c2:
            st.button(
                "🔬 Open Full Forensic Evidence Workbench & SQLite Ledger",
                on_click=_jump_to_section,
                args=(SECTION_EVIDENCE,),
                width="stretch",
                key="jump_evd_from_img",
            )


# ==============================================================================
# SECTION 5: FORENSIC EVIDENCE (XAI WORKBENCH + SQLITE AUDIT TRAIL)
# ==============================================================================
def render_forensic_evidence_section() -> None:
    st.subheader("🔬 Forensic Evidence Workbench & Cryptographic Audit Ledger")
    tab_xai_deep, tab_audit_ledger = st.tabs(
        [
            "🧠 Active Examination XAI Workbench (Grad-CAM / 2D FFT / ELA)",
            "📜 SQLite Forensic Audit Ledger & Raw File Hash Verifier",
        ]
    )

    with tab_xai_deep:
        img_res = st.session_state.get("analysis_results")
        vid_frame_res = st.session_state.get("vid_frame_xai_result")

        active_source_choice = None
        if img_res is not None and vid_frame_res is not None:
            active_source_choice = st.radio(
                "Select Active Examination to Inspect:",
                ["Last Analyzed Image", "Last Inspected Video Frame"],
                horizontal=True,
                key="ev_xai_src_radio",
            )
        elif img_res is not None:
            active_source_choice = "Last Analyzed Image"
        elif vid_frame_res is not None:
            active_source_choice = "Last Inspected Video Frame"

        target_res = img_res if active_source_choice == "Last Analyzed Image" else vid_frame_res

        if target_res is None:
            st.info(
                "No image or video frame XAI result is active in this session yet. "
                "Analyze an image in **🖼️ Image Analysis**, inspect a frame in **🎥 Video Analysis**, "
                "or run a quick benchmark sample inspection below:"
            )
            demo_sample = REPO_ROOT / "data" / "demo_samples" / "deepfake_diffusion_ai_face.jpg"
            if demo_sample.exists() and st.button("🧪 Run Quick XAI Inspection on Diffusion Sample", key="quick_xai_demo_btn", type="primary"):
                raw_b = demo_sample.read_bytes()
                pil_demo = Image.open(io.BytesIO(raw_b)).convert("RGB")
                with st.spinner("Running Dual-Domain XAI Inspection..."):
                    res_demo = pipeline.analyze_image(pil_demo)
                    db.log_examination(
                        res_demo,
                        image_source_name=demo_sample.name,
                        device_str=device.type.upper(),
                        raw_file_bytes=raw_b,
                        file_size_bytes=len(raw_b),
                        media_type="image",
                        media_dimensions=f"{pil_demo.width}x{pil_demo.height}",
                        actor="streamlit_forensic_evidence_quickrun",
                    )
                    st.session_state["analysis_results"] = res_demo
                    st.session_state["analyzed_image_source"] = demo_sample.name
                    st.session_state["analyzed_raw_sha256"] = compute_bytes_sha256(raw_b)
                    st.rerun()
        else:
            st.caption(
                f"Active Case: `{target_res.get('case_id', 'Frame-On-Demand')}` | "
                f"Outcome: `{target_res.get('prediction_outcome', 'N/A')}` | "
                f"Calibrated P(Manipulated): `{float(target_res.get('fake_probability', 0.5)):.4f}`"
            )
            multi_faces = target_res.get("multi_face_reports", [])
            active_rep = get_active_face_report(target_res, key_prefix="evd_sec")
            render_unified_forensic_panel(target_res, active_rep)

            x_t1, x_t2, x_t3, x_t4, x_t5 = st.tabs(
                [
                    "🧠 Spatial Grad-CAM Details",
                    "📊 2D FFT Radial Spectrum Comparison",
                    "🔬 Spatial SRM Noise Residuals",
                    "🔍 Multi-Quality ELA & Limitations",
                    "📄 Compile Official PDF Report",
                ]
            )
            with x_t1:
                render_spatial_gradcam_tab(active_rep, multi_faces)
            with x_t2:
                render_spectral_fft_tab(active_rep)
            with x_t3:
                render_srm_tab(active_rep)
            with x_t4:
                render_ela_tab(active_rep, target_res)
            with x_t5:
                st.write("Generates a multi-page ReportLab Forensic Certificate including raw vs crop SHA-256 digests, decoupled indicators, radial FFT spectra, and `audit_events`.")
                if st.button("📑 Compile Evidence Tab PDF Certificate", key="ev_gen_pdf_btn", width="stretch"):
                    out_pdf_name = f"Forensic_Certificate_{target_res.get('case_id', 'LIVE')}.pdf"
                    pdf_bytes = _compile_and_log_image_pdf(target_res, out_pdf_name)
                    st.session_state["ev_pdf_ready_path"] = str(_get_pdf_report_path())
                    st.success(f"PDF compiled ({len(pdf_bytes):,} bytes) and logged.")
                if st.session_state.get("ev_pdf_ready_path") and Path(st.session_state["ev_pdf_ready_path"]).exists():
                    st.download_button(
                        label="⬇️ Download Evidence PDF Certificate",
                        data=Path(st.session_state["ev_pdf_ready_path"]).read_bytes(),
                        file_name=f"Forensic_Certificate_{target_res.get('case_id', 'LIVE')}.pdf",
                        mime="application/pdf",
                        width="stretch",
                        key="ev_dl_pdf_btn",
                    )

    with tab_audit_ledger:
        st.markdown("#### 📜 Software Forensic Audit Ledger & Immutable Event Log")
        st.info(f"⚖️ **{CHAIN_OF_CUSTODY_DISCLAIMER}**")

        f_col1, f_col2 = st.columns([1.2, 1.0])
        with f_col1:
            include_archived = st.checkbox(
                "Include archived examination records",
                value=False,
                key="hist_include_archived_chk",
            )
        with f_col2:
            media_filter = st.selectbox(
                "Filter Media Type:",
                ["All Media", "image", "video"],
                key="hist_media_filter_sel",
            )

        filter_arg = None if media_filter == "All Media" else media_filter
        history_records = db.get_history(limit=200, include_archived=include_archived, media_type=filter_arg)
        schema_info = db.get_schema_info()

        if history_records:
            df_history = pd.DataFrame(history_records)
            display_cols = [
                "case_id",
                "evidence_id",
                "media_type",
                "timestamp",
                "image_name",
                "media_dimensions",
                "file_size_bytes",
                "raw_file_sha256",
                "prediction_outcome",
                "verdict_tier",
                "fake_probability",
                "decision_threshold",
                "confidence_percent",
                "processing_status",
                "latency_ms",
                "is_archived",
            ]
            available_cols = [c for c in display_cols if c in df_history.columns]

            c_tot, c_manip, c_auth, c_evts = st.columns(4)
            c_tot.metric(
                "Visible Examinations",
                len(df_history),
                delta=f"{schema_info['archived_examinations']} archived",
                delta_color="off",
            )
            manip_count = len(
                df_history[
                    df_history["prediction_outcome"].astype(str).str.contains("MANIPULATED", case=False, na=False)
                    | df_history["verdict_tier"].astype(str).str.contains("Manipulated", case=False, na=False)
                ]
            )
            auth_count = len(
                df_history[
                    df_history["prediction_outcome"].astype(str).str.contains("AUTHENTIC", case=False, na=False)
                    | df_history["verdict_tier"].astype(str).str.contains("Authentic", case=False, na=False)
                ]
            )
            c_manip.metric("Flagged Manipulated", manip_count)
            c_auth.metric("Verified Authentic", auth_count)
            c_evts.metric("Immutable Audit Events", schema_info["total_audit_events"])

            st.dataframe(df_history[available_cols], width="stretch", hide_index=True)

            # Per-Case Hash & Immutable Lifecycle Inspector
            case_options = [r["case_id"] for r in history_records]
            selected_case_id = st.selectbox(
                "🔎 Inspect Case Hashes & Immutable Audit Lifecycle (`audit_events`):",
                case_options,
                key="hist_inspect_case_sel",
            )
            sel_record = next((r for r in history_records if r["case_id"] == selected_case_id), None)
            if sel_record is not None:
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown(
                        f"**Case ID:** `{sel_record.get('case_id')}` | **Evidence ID:** `{sel_record.get('evidence_id')}`\n\n"
                        f"**Media Type:** `{sel_record.get('media_type', 'image')}` | **Dimensions:** `{sel_record.get('media_dimensions', 'unknown')}`\n\n"
                        f"**Size:** `{int(sel_record.get('file_size_bytes', 0)):,}` bytes | "
                        f"**Threshold tau:** `{float(sel_record.get('decision_threshold', 0.5)):.2f}` | "
                        f"**Temp T:** `{float(sel_record.get('calibrated_temperature', 1.0)):.4f}`"
                    )
                with ic2:
                    st.code(
                        f"Raw Uploaded SHA-256 : {sel_record.get('raw_file_sha256', 'N/A')}\n"
                        f"Aligned Crop SHA-256 : {sel_record.get('image_sha256', 'N/A')}",
                        language="text",
                    )

                case_events = _fetch_case_events(selected_case_id, sel_record.get("evidence_id"))
                if case_events:
                    df_evts = pd.DataFrame(case_events)[
                        ["event_id", "event_type", "event_timestamp_utc", "sha256_at_event", "actor", "details_json"]
                    ]
                    st.dataframe(df_evts, width="stretch", hide_index=True)
                else:
                    st.info("No lifecycle events recorded for this legacy case prior to Phase 8 migration.")

                # Live Raw Media Tamper Verification Tool
                with st.expander("🔐 Verify Raw Uploaded File Integrity Against Stored SHA-256", expanded=False):
                    verify_upload = st.file_uploader(
                        f"Upload candidate file to verify byte-for-byte match against `{selected_case_id}`:",
                        key="verify_tamper_uploader",
                    )
                    if verify_upload is not None:
                        v_res = db.verify_evidence_integrity(selected_case_id, raw_bytes=verify_upload.getvalue())
                        if v_res["integrity_verified"]:
                            st.success(
                                f"✅ INTEGRITY VERIFIED (`MATCH`): Candidate SHA-256 `{v_res['candidate_sha256']}` "
                                f"matches recorded raw media digest identically ({v_res['candidate_size_bytes']:,} bytes)."
                            )
                        else:
                            st.error(
                                f"🚨 INTEGRITY CHECK `{v_res['status']}`: Candidate SHA-256 `{v_res['candidate_sha256']}` "
                                f"does NOT match stored digest `{v_res['stored_raw_file_sha256']}`."
                            )

            col_export, col_clear = st.columns([1.6, 1.4])
            with col_export:
                csv_data = df_history.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Examination Ledger (CSV)",
                    data=csv_data,
                    file_name="forensic_examination_audit_trail.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with col_clear:
                confirm_archive = st.checkbox(
                    "Confirm archiving active examinations (`audit_events` remain immutable)",
                    value=False,
                    key="confirm_archive_hist_chk",
                )
                if st.button(
                    "🗄️ Archive Active History",
                    key="archive_hist_btn",
                    disabled=not confirm_archive,
                    width="stretch",
                ):
                    archived_n = db.clear_history(confirm=True, archive_only=True, actor="streamlit_operator")
                    st.success(f"Archived {archived_n} active record(s). Immutable audit events preserved.")
                    st.rerun()
        else:
            st.info("No active examination records found in `data/forensic_history.db`. Run an image or video inspection to log your first case.")


# ==============================================================================
# SECTION 6: MODEL INFORMATION
# ==============================================================================
def render_model_information_section() -> None:
    st.subheader("ℹ️ Model Architecture, Provenance & Scientific Limitations")

    model = pipeline.model
    total_params = sum(p.numel() for p in model.parameters())
    spatial_backbone_params = sum(p.numel() for p in model.spatial_stream.features.parameters())
    spatial_proj_params = sum(p.numel() for p in model.spatial_stream.proj.parameters())
    freq_params = sum(p.numel() for p in model.frequency_stream.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters()) if model.fusion is not None else 0
    cls_params = sum(p.numel() for p in model.classifier.parameters())

    eval_data = load_evaluation_data()
    ckpt = eval_data.get("checkpoint", {})
    cal = eval_data.get("calibration", {})
    audit = eval_data.get("dataset_audit", {})
    ext = eval_data.get("external_benchmark", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Model Parameters", f"{total_params:,}")
    m2.metric("Spatial Stream (Backbone + Proj)", f"{spatial_backbone_params + spatial_proj_params:,}")
    m3.metric("Frequency + Fusion Params", f"{freq_params + fusion_params:,}")
    m4.metric("Classifier Head Params", f"{cls_params:,}")

    st.markdown("### 🧩 Dual-Domain Neural Architecture (`DualDomainSentinelNet`)")
    st.markdown(
        f"""
| Subsystem | Implementation Details | Parameter Count |
| :--- | :--- | :---: |
| **Spatial Stream Features** | `torchvision` EfficientNet-B0 convolutional trunk + AdaptiveAvgPool2d | `{spatial_backbone_params:,}` |
| **Spatial Projection** | Linear (`1280 -> 512`) + LayerNorm + ReLU | `{spatial_proj_params:,}` |
| **Frequency Stream** | 3 Fixed 5x5 SRM High-Pass Filters + 2D FFT Log-Magnitude 4-Stage ConvNet -> `256`-d | `{freq_params:,}` |
| **Cross-Domain Fusion** | 4-Head `nn.MultiheadAttention` (`embed_dim=256`, Queries=Spatial, Keys/Values=Spectral) | `{fusion_params:,}` |
| **Classification Head** | MLP `256 -> 128 -> 32 -> 1` raw logit (plus Sigmoid/Temperature scaling at inference) | `{cls_params:,}` |
| **Training Objective** | `LabelSmoothingBCELoss(smoothing=0.06)` (`BinaryFocalLoss` implemented cleanly in `loss.py`) | Total: **`{total_params:,}`** |
"""
    )

    st.markdown("### ⚡ Runtime Hardware & Memory Telemetry")
    rt_mem = get_runtime_memory_telemetry(getattr(pipeline, "device", device))
    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric("Active Execution Device", rt_mem["device"].upper(), delta="Auto MPS->CUDA->CPU fallback", delta_color="off")
    hm2.metric("Process Memory (CPU RSS)", f"{rt_mem['cpu_rss_mb']:.1f} MB", delta=rt_mem["memory_backend"], delta_color="off")
    hm3.metric("Accelerator Allocated", f"{rt_mem['accelerator_alloc_mb']:.1f} MB", delta=f"Reserved: {rt_mem['accelerator_reserved_mb']:.1f} MB", delta_color="off")
    hm4.metric("Inference Engine", "Batched", delta="Multi-face & Video mini-batch", delta_color="off")

    st.markdown("### 📦 Provenance, Runtime & Calibration State")
    pol = cal.get("uncertainty_policy", {})
    temp_val = cal.get("temperature_scaling", {}).get("temperature", 1.0)
    avail_ext = sum(1 for v in (ext.get("datasets") or {}).values() if v.get("available"))
    st.markdown(
        f"""
- **Production Checkpoint:** `{ckpt.get('path', 'checkpoints/best_model.pth')}` (`{ckpt.get('size_mb', 0.0)} MB`) — SHA-256: `{ckpt.get('sha256', 'N/A')}`
- **Calibration Policy (Validation-Fit Only):** `T = {float(temp_val):.4f}`, Selected Threshold `tau = {float(pol.get('decision_threshold', 0.50)):.2f}`, Uncertainty Band `[{float(pol.get('uncertain_low', 0.40)):.3f}, {float(pol.get('uncertain_high', 0.60)):.3f}]` (`zero_test_set_tuning = True`)
- **Group-Disjoint Dataset Audit:** `{audit.get('total_image_count', 0)}` total images across `{audit.get('source_group_analysis', {}).get('total_source_groups', 'N/A')}` source groups (`0` leaked source groups via `StratifiedGroupKFold`, `0` exact duplicates)
- **External Benchmarks Available Locally:** `{avail_ext}/3` (FaceForensics++, Celeb-DF, DFDC cleanly checked without fabricated fallback metrics)
- **Active Execution Environment:** Python `{platform.python_version()}`, PyTorch `{torch.__version__}`, OpenCV `{cv2.__version__}`, Accelerator `{rt_mem['device'].upper()}` (`{platform.platform()}`)
"""
    )

    st.markdown("### ⚠️ Known Scientific Limitations & Honest Disclosures")
    st.warning(
        "1. **Probabilistic Evidence Only:** Deepfake classifiers quantify statistical patterns and are not infallible detectors against unseen generators or adversarial re-encoding.\n"
        "2. **XAI Non-Proof Caveat:** Grad-CAM focuses on model attention regions; 2D FFT spikes and JPEG ELA residuals also respond to benign image sharpening, resizing, camera ISP processing, and platform re-compression (WhatsApp/Instagram).\n"
        "3. **Video Frame-Level Inference (`temporal_modeling=False`):** The video analyzer runs frame-by-frame spatial + 2D FFT evaluation with multi-frame quorum aggregation; it does not evaluate inter-frame optical flow, lip-sync audio, or 3D spatiotemporal transformers.\n"
        "4. **Training Set Domain Bound:** The production dataset (`2,400` images) is derived from bounded local portrait sources (`15` source groups); out-of-distribution generalization requires external evaluation on downloaded FaceForensics++ / Celeb-DF / DFDC splits.\n"
        f"5. **Audit Log Boundary:** {CHAIN_OF_CUSTODY_DISCLAIMER}"
    )

    st.markdown("### 👨‍💻 Academic & Capstone Metadata")
    st.markdown(
        "- **Lead Developer:** Antriksh Mishra\n"
        "- **Degree Program:** Final Year B.Tech — Computer Science & Engineering (Computer Vision & Deep Learning)\n"
        "- **System Version:** Deepfake Forensic Sentinel (`DualDomainSentinel-v1.0`)"
    )


# Route cleanly to active section
if active_section == SECTION_OVERVIEW:
    render_overview_section()
elif active_section == SECTION_IMAGE:
    render_image_analysis_section()
elif active_section == SECTION_VIDEO:
    render_video_analysis_view(pipeline, db=db, device_str=device.type.upper())
elif active_section == SECTION_PERFORMANCE:
    render_performance_dashboard()
elif active_section == SECTION_EVIDENCE:
    render_forensic_evidence_section()
elif active_section == SECTION_INFO:
    render_model_information_section()

# Persistent Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.88rem; padding: 8px;'>"
    "🛡️ <b>Deepfake Forensic Sentinel</b> &bull; Developed &amp; Researched by <b>Antriksh Mishra</b> &bull; "
    "Final Year Computer Vision Capstone &bull; © 2026"
    "</div>",
    unsafe_allow_html=True,
)
