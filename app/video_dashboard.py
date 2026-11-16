"""
Video Deepfake Forensic Detection Dashboard Module.
Renders video upload controls, configurable frame sampling, overall calibrated
assessment, interactive color-coded timeline (Green/Red/Yellow), clickable
timestamp frame inspection, confidence distribution plots, and downloadable
frame-level forensic reports.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.core.database import ForensicDatabase, compute_stream_sha256

try:
    from app.report_generator import generate_video_pdf_report
except (ImportError, ModuleNotFoundError):
    from report_generator import generate_video_pdf_report
from src.evaluation.calibration import (
    OUTCOME_AUTHENTIC,
    OUTCOME_MANIPULATED,
    OUTCOME_UNCERTAIN,
)
from src.video.video_analyzer import (
    DEFAULT_MAX_VIDEO_SIZE_MB,
    FRAME_OUTCOME_NO_FACE,
    OUTCOME_HEX_MAP,
    VideoForensicAnalyzer,
    build_video_forensic_report,
    format_video_forensic_markdown,
)


def _timeline_emoji(color_name: str) -> str:
    return {
        "green": "🟢",
        "red": "🔴",
        "yellow": "🟡",
        "gray": "⚪",
    }.get(color_name, "⚪")


def plot_video_timeline(
    timeline: List[Dict[str, Any]],
    calibration_meta: Dict[str, Any],
    selected_pos: Optional[int] = None,
) -> plt.Figure:
    """
    Renders a color-coded Matplotlib timeline plot of calibrated frame-level probabilities:
      Green  = AUTHENTIC
      Red    = MANIPULATED
      Yellow = UNCERTAIN / NEEDS MANUAL REVIEW
      Gray   = NO FACE DETECTED
    """
    fig, ax = plt.subplots(figsize=(10.5, 3.6))

    tau = float(calibration_meta.get("decision_threshold", 0.50))
    u_low = float(calibration_meta.get("uncertain_low", 0.40))
    u_high = float(calibration_meta.get("uncertain_high", 0.60))

    # Ambiguity band & threshold line
    ax.axhspan(u_low, u_high, color="#fde68a", alpha=0.32, label=f"Uncertainty Band [{u_low:.3f}, {u_high:.3f}]")
    ax.axhline(tau, color="#475569", linestyle="--", linewidth=1.2, label=f"Threshold tau = {tau:.2f}")

    face_pts = [r for r in timeline if r.get("has_face") and r.get("fake_probability") is not None]
    if face_pts:
        ts_face = [r["timestamp_sec"] for r in face_pts]
        ps_face = [r["fake_probability"] for r in face_pts]
        ax.plot(ts_face, ps_face, color="#94a3b8", linewidth=1.2, alpha=0.75, zorder=2)

    outcome_groups = [
        (OUTCOME_AUTHENTIC, OUTCOME_HEX_MAP["green"], "o", "Authentic (Green)"),
        (OUTCOME_MANIPULATED, OUTCOME_HEX_MAP["red"], "o", "Manipulated (Red)"),
        (OUTCOME_UNCERTAIN, OUTCOME_HEX_MAP["yellow"], "D", "Uncertain / Review (Yellow)"),
        (FRAME_OUTCOME_NO_FACE, OUTCOME_HEX_MAP["gray"], "x", "No Face Detected (Gray)"),
    ]
    for outcome_key, hex_col, marker, label in outcome_groups:
        grp = [r for r in timeline if r.get("prediction_outcome") == outcome_key]
        if not grp:
            continue
        xs = [r["timestamp_sec"] for r in grp]
        ys = [r["fake_probability"] if r.get("fake_probability") is not None else 0.03 for r in grp]
        ax.scatter(
            xs,
            ys,
            c=hex_col,
            marker=marker,
            s=68 if marker != "x" else 55,
            edgecolors="#0f172a" if marker != "x" else None,
            linewidths=0.7,
            zorder=4,
            label=label,
        )

    if selected_pos is not None and 0 <= selected_pos < len(timeline):
        sel_r = timeline[selected_pos]
        sel_x = sel_r["timestamp_sec"]
        sel_y = sel_r["fake_probability"] if sel_r.get("fake_probability") is not None else 0.03
        ax.axvline(sel_x, color="#1d4ed8", linestyle=":", linewidth=1.5, alpha=0.85, zorder=3)
        ax.scatter(
            [sel_x],
            [sel_y],
            s=190,
            facecolors="none",
            edgecolors="#1d4ed8",
            linewidths=2.2,
            zorder=5,
            label=f"Inspected ({sel_r['timestamp_label']})",
        )

    ax.set_ylim(-0.04, 1.16)
    if timeline:
        max_t = max(float(r["timestamp_sec"]) for r in timeline)
        ax.set_xlim(-0.04 * max(max_t, 1.0), max(max_t, 0.5) + 0.04 * max(max_t, 1.0))
    ax.set_xlabel("Video Timestamp (seconds)", fontsize=9.5)
    ax.set_ylabel("Calibrated P(Manipulated)", fontsize=9.5)
    ax.set_title("Frame-Level Deepfake Timeline (Independent Frame Inference)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=7.8, ncol=3, framealpha=0.94)
    fig.tight_layout()
    return fig


def plot_confidence_distribution(
    results: Dict[str, Any],
) -> plt.Figure:
    """Renders probability histogram + frame outcome counts side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))

    conf_dist = results.get("confidence_distribution", {})
    probs = conf_dist.get("fake_probabilities", [])
    cal_meta = results.get("calibration_metadata", {})
    tau = float(cal_meta.get("decision_threshold", 0.50))
    u_low = float(cal_meta.get("uncertain_low", 0.40))
    u_high = float(cal_meta.get("uncertain_high", 0.60))

    if probs:
        n_hist, _, _ = ax1.hist(probs, bins=np.linspace(0.0, 1.0, 15), color="#3b82f6", alpha=0.78, edgecolor="#1e3a8a")
        ax1.axvspan(u_low, u_high, color="#fde68a", alpha=0.35, label="Uncertainty Band")
        ax1.axvline(tau, color="#dc2626", linestyle="--", linewidth=1.3, label=f"tau={tau:.2f}")
        ax1.set_ylim(0, max(float(np.max(n_hist)), 1.0) * 1.28)
        ax1.legend(fontsize=8, loc="upper left", framealpha=0.94)
    else:
        ax1.text(0.5, 0.5, "No face-detected frames", ha="center", va="center", color="#64748b")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("Calibrated Frame P(Manipulated)", fontsize=9)
    ax1.set_ylabel("Frame Count", fontsize=9)
    ax1.set_title("Frame Probability Distribution", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    categories = ["Authentic\n(Green)", "Manipulated\n(Red)", "Uncertain\n(Yellow)", "No Face\n(Gray)"]
    counts = [
        int(results.get("authentic_frame_count", 0)),
        int(results.get("manipulated_frame_count", 0)),
        int(results.get("uncertain_frame_count", 0)),
        int(results.get("frames_without_faces", 0)),
    ]
    colors = [
        OUTCOME_HEX_MAP["green"],
        OUTCOME_HEX_MAP["red"],
        OUTCOME_HEX_MAP["yellow"],
        OUTCOME_HEX_MAP["gray"],
    ]
    bars = ax2.bar(categories, counts, color=colors, edgecolor="#1e293b", alpha=0.9)
    for b in bars:
        h = b.get_height()
        ax2.annotate(
            f"{int(h)}",
            xy=(b.get_x() + b.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_ylim(0, max(counts + [1]) * 1.22)
    ax2.set_ylabel("Sampled Frames", fontsize=9)
    ax2.set_title("Frame Outcome Breakdown", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def _on_video_report_download(
    db: Optional[ForensicDatabase],
    case_id: Optional[str],
    evidence_id: Optional[str],
    report_format: str,
    report_bytes: bytes,
    output_name: str,
) -> None:
    """Callback triggered when a user downloads a structured video forensic report."""
    if db is not None:
        db.record_report_generation(
            case_id=case_id,
            evidence_id=evidence_id,
            report_format=report_format,
            report_bytes=report_bytes,
            output_name=output_name,
            actor="streamlit_video_dashboard",
        )


def render_video_analysis_view(
    pipeline: Any,
    max_file_size_mb: float = DEFAULT_MAX_VIDEO_SIZE_MB,
    db: Optional[ForensicDatabase] = None,
    device_str: str = "CPU",
) -> None:
    """Main Streamlit UI for Phase 6 Video Deepfake Detection."""
    st.markdown("### 🎥 Video Forensic Examination (Frame-Level Dual-Domain Analysis)")
    st.info(
        "**Methodological Scope:** Sampled video frames are analyzed **independently** using the existing "
        "`DualDomainSentinelNet` (RGB spatial + 2D FFT spectral) with validation-calibrated temperature scaling "
        "and the 3-outcome policy (`AUTHENTIC`, `MANIPULATED`, `UNCERTAIN / NEEDS MANUAL REVIEW`). "
        "No temporal neural network (3D CNN / optical flow / transformer) is active."
    )

    with st.expander("⚙️ Sampling & Calibration Settings", expanded=True):
        c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns([1.4, 1.2, 1.2, 1.2])
        with c_cfg1:
            sampling_label = st.selectbox(
                "Frame Sampling Strategy",
                ["Frames Per Second (FPS)", "Every N Frames (Stride)", "Uniform Timeline Spread"],
                index=0,
                key="vid_sampling_strategy",
            )
            mode_map = {
                "Frames Per Second (FPS)": "fps",
                "Every N Frames (Stride)": "interval",
                "Uniform Timeline Spread": "uniform",
            }
            sample_mode = mode_map[sampling_label]
        with c_cfg2:
            if sample_mode == "fps":
                sample_fps = float(
                    st.slider("Sample Rate (FPS)", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key="vid_sample_fps")
                )
                sample_every_n = 15
            elif sample_mode == "interval":
                sample_every_n = int(
                    st.number_input("Decode Every Nth Frame", min_value=1, max_value=120, value=15, step=1, key="vid_sample_stride")
                )
                sample_fps = 1.0
            else:
                sample_fps = 1.0
                sample_every_n = 15
                st.caption("Uniformly samples `max_frames` linearly across the video duration.")
        with c_cfg3:
            max_frames = int(
                st.slider("Max Sampled Frames", min_value=4, max_value=60, value=24, step=2, key="vid_max_frames")
            )
        with c_cfg4:
            use_default_thresh = st.checkbox(
                "Use 0.50 Default Threshold",
                value=False,
                help="By default uses the validation-calibrated threshold from calibration_results.json.",
                key="vid_use_default_thresh",
            )
            center_crop_fallback = st.checkbox(
                "Center-crop if no Haar face",
                value=False,
                help="Keep disabled to flag videos without visible faces as NO FACES DETECTED.",
                key="vid_center_crop_fallback",
            )

    uploaded_video = st.file_uploader(
        f"Upload Video File (MP4, AVI, MOV — up to {int(max_file_size_mb)} MB)",
        type=["mp4", "avi", "mov"],
        key="vid_file_uploader",
    )

    if uploaded_video is None:
        st.info("👆 Upload an `.mp4`, `.avi`, or `.mov` video clip above to start frame-level forensic examination.")
        return

    file_size_bytes = int(getattr(uploaded_video, "size", 0) or 0)
    upload_sig = f"{uploaded_video.name}:{file_size_bytes}"
    if st.session_state.get("vid_last_upload_sig") != upload_sig:
        st.session_state["vid_last_upload_sig"] = upload_sig
        st.session_state["vid_results"] = None
        st.session_state["vid_selected_pos"] = 0
        st.session_state["vid_frame_xai_result"] = None
        st.session_state["vid_frame_xai_pos"] = None
        st.session_state["vid_pdf_bytes"] = None
        if db is not None:
            stream_sha, stream_bytes = compute_stream_sha256(uploaded_video)
            intake = db.record_media_upload(
                filename=uploaded_video.name,
                raw_file_sha256=stream_sha,
                file_size_bytes=stream_bytes or file_size_bytes,
                media_type="video",
                media_dimensions="pending_container_probe",
                actor="streamlit_video_dashboard",
            )
            st.session_state["vid_evidence_id"] = intake["evidence_id"]
            st.session_state["vid_raw_sha256"] = intake["raw_file_sha256"]
        else:
            st.session_state["vid_evidence_id"] = None
            st.session_state["vid_raw_sha256"] = None

    col_info, col_btn = st.columns([2, 1])
    with col_info:
        st.write(
            f"**Selected Video:** `{uploaded_video.name}` "
            f"(`{file_size_bytes / (1024 * 1024):.2f} MB`)"
        )
    with col_btn:
        run_vid_btn = st.button("🚀 Analyze Video Frames", type="primary", width="stretch", key="vid_run_btn")

    if run_vid_btn:
        suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
        tmp_path: Optional[str] = None
        prog_bar = st.progress(0.0, text="Preparing video container...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_video.getbuffer())
                tmp_path = tmp.name

            analyzer = VideoForensicAnalyzer(pipeline=pipeline, max_file_size_mb=max_file_size_mb)
            threshold_arg = 0.50 if use_default_thresh else None

            def _ui_progress(frac: float, msg: str) -> None:
                prog_bar.progress(float(np.clip(frac, 0.0, 1.0)), text=msg)

            res = analyzer.analyze_video(
                video_path=tmp_path,
                sample_mode=sample_mode,
                sample_fps=sample_fps,
                sample_every_n_frames=sample_every_n,
                max_frames=max_frames,
                threshold=threshold_arg,
                allow_center_crop_fallback=center_crop_fallback,
                progress_callback=_ui_progress,
            )
            # Preserve the original uploaded filename in displayed output rather than the temp name
            res["filename"] = uploaded_video.name
            if isinstance(res.get("file_metadata"), dict):
                res["file_metadata"]["filename"] = uploaded_video.name

            if db is not None:
                db.log_video_examination(
                    res,
                    device_str=device_str,
                    evidence_id=st.session_state.get("vid_evidence_id"),
                    upload_already_logged=bool(st.session_state.get("vid_evidence_id")),
                    actor="streamlit_video_dashboard",
                )

            st.session_state["vid_results"] = res
            st.session_state["vid_selected_pos"] = 0
            st.session_state["vid_frame_xai_result"] = None
            st.session_state["vid_frame_xai_pos"] = None
            st.session_state["vid_pdf_bytes"] = None
            prog_bar.empty()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    results = st.session_state.get("vid_results")
    if results is None:
        return

    st.markdown("---")

    # Handle validation / corruption errors gracefully
    if results.get("status") == "error":
        st.error(f"❌ {results.get('error', 'Failed to process video file.')}")
        st.caption(results.get("methodology_disclaimer", ""))
        return

    # 1. Overall Video Assessment Banner
    verdict_class = results.get("verdict_class", "verdict-inconclusive")
    verdict_badge = results.get("verdict_badge", results.get("overall_assessment", "VIDEO ASSESSMENT"))
    verdict_exp = results.get("verdict_explanation", "")
    agg_prob = results.get("aggregated_fake_probability")
    agg_prob_str = f"{agg_prob:.4f}" if agg_prob is not None else "N/A"
    agg_conf = float(results.get("aggregated_confidence_percent", 0.0))
    cal_meta = results.get("calibration_metadata", {})

    st.markdown(
        f'<div class="{verdict_class}">{verdict_badge}<br/>'
        f'<span style="font-size: 1.02rem; font-weight: 400;">'
        f'Overall Assessment: <b>{results.get("overall_assessment", "N/A")}</b> &nbsp;|&nbsp; '
        f'Aggregated Confidence: <b>{agg_conf:.2f}%</b> &nbsp;|&nbsp; '
        f'Trimmed Mean P(Manipulated): <b>{agg_prob_str}</b>'
        f'</span>'
        f'<div style="font-size: 0.80rem; font-weight: 400; margin-top: 4px; opacity: 0.9;">'
        f'Calibration: T={cal_meta.get("temperature", 1.0):.4f} | tau={cal_meta.get("decision_threshold", 0.50):.2f} '
        f'| Band=[{cal_meta.get("uncertain_low", 0.40):.3f}, {cal_meta.get("uncertain_high", 0.60):.3f}] '
        f'| Scope: <code>{results.get("analysis_scope", "frame_level_inference")}</code> (temporal_modeling={results.get("temporal_modeling", False)})'
        f'</div>'
        f'<div style="font-size: 0.88rem; font-weight: 400; margin-top: 6px; opacity: 0.94;">{verdict_exp}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="disclaimer-box">'
        f'<b>🛡️ Aggregation Safeguard & Methodological Scope:</b><br/>'
        f'<b>Aggregation Rule:</b> {results.get("aggregation_method", "")}<br/>'
        f'<b>Scope Notice:</b> {results.get("methodology_disclaimer", "")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if results.get("isolated_spike_suppressed"):
        st.warning(
            "🔎 **Single-Frame Safeguard Activated:** Only 1 isolated manipulated frame was observed. "
            "Per policy, a single suspicious frame never marks an entire video as manipulated; click its red marker below to inspect it."
        )

    # 2. Required Video Telemetry Metrics (2 clean rows)
    r1_1, r1_2, r1_3, r1_4, r1_5 = st.columns(5)
    r1_1.metric("Video Filename", results.get("filename", "N/A"))
    r1_2.metric("Duration", f"{results.get('duration_formatted', '00:00.00')} ({results.get('duration_sec', 0.0):.2f}s)")
    r1_3.metric("Frame Count", f"{results.get('frame_count', 0):,}", help=f"Reported FPS: {results.get('fps', 0.0):.2f}")
    r1_4.metric("Processed Frames", f"{results.get('processed_frames', 0)}")
    r1_5.metric(
        "Detected Faces",
        f"{results.get('detected_faces', 0)}",
        delta=f"{results.get('frames_with_faces', 0)} frames w/ face",
        delta_color="off",
    )

    r2_1, r2_2, r2_3, r2_4, r2_5 = st.columns(5)
    r2_1.metric("Authentic Frames", f"{results.get('authentic_frame_pct', 0.0):.1f}%", delta=f"{results.get('authentic_frame_count', 0)} frames", delta_color="off")
    r2_2.metric("Manipulated Frames", f"{results.get('manipulated_frame_pct', 0.0):.1f}%", delta=f"{results.get('manipulated_frame_count', 0)} frames", delta_color="off")
    r2_3.metric("Uncertain Frames", f"{results.get('uncertain_frame_pct', 0.0):.1f}%", delta=f"{results.get('uncertain_frame_count', 0)} frames", delta_color="off")
    r2_4.metric("P90 / Median P(Fake)", f"{results.get('p90_fake_probability') or 0.0:.3f} / {results.get('median_fake_probability') or 0.0:.3f}")
    r2_5.metric("Processing Latency", f"{results.get('inference_time_ms', 0.0):.1f} ms", delta=f"{results.get('processing_fps', 0.0):.1f} sampled fps", delta_color="off")

    file_meta = results.get("file_metadata", {})
    vid_case_id = results.get("case_id", "N/A")
    vid_evd_id = results.get("evidence_id", st.session_state.get("vid_evidence_id", "N/A"))
    st.caption(
        f"🔐 **Raw File SHA-256:** `{results.get('sha256', 'N/A')}` &bull; "
        f"**Case ID:** `{vid_case_id}` &bull; **Evidence ID:** `{vid_evd_id}` &bull; "
        f"**Resolution:** `{file_meta.get('width', 0)}x{file_meta.get('height', 0)}` &bull; "
        f"**FPS:** `{results.get('fps', 0.0):.2f}` &bull; "
        f"**Size:** `{file_meta.get('file_size_mb', 0.0):.2f} MB`"
    )
    vid_tel = results.get("performance_telemetry", {})
    if vid_tel:
        fb_note = " | CPU fallback triggered" if vid_tel.get("cpu_fallback_triggered") else ""
        st.caption(
            f"⚡ **Batch & Memory Telemetry:** Device `{str(vid_tel.get('device', 'cpu')).upper()}` &bull; "
            f"Batch Size: `{vid_tel.get('batch_size', 16)}` &bull; "
            f"Neural Forward: `{vid_tel.get('neural_inference_ms', 0.0):.1f} ms` "
            f"(`{vid_tel.get('neural_throughput_fps', 0.0):.1f} face/s`) &bull; "
            f"CPU RAM: `{vid_tel.get('cpu_rss_mb', 0.0):.1f} MB` &bull; "
            f"Accel Alloc: `{vid_tel.get('accelerator_alloc_mb', 0.0):.1f} MB`{fb_note}"
        )

    if results.get("status") == "no_faces_detected":
        st.warning(
            "⚠️ **No Detectable Human Faces Found:** None of the sampled frames yielded a verifiable face detection. "
            "Try increasing `Max Sampled Frames` or check framing/lighting."
        )

    # 3. Interactive Timeline & Clickable Frame Inspection
    timeline = results.get("timeline", [])
    if timeline:
        st.markdown("---")
        st.subheader("🕒 Interactive Forensic Timeline")
        st.caption(
            "🟢 **Green** = Authentic | 🔴 **Red** = Manipulated | 🟡 **Yellow** = Uncertain / Needs Review | ⚪ **Gray** = No Face Detected. "
            "**Click any timestamp button below** to inspect that exact video frame."
        )

        selected_pos = int(st.session_state.get("vid_selected_pos", 0))
        selected_pos = max(0, min(selected_pos, len(timeline) - 1))

        fig_tl = plot_video_timeline(timeline, cal_meta, selected_pos=selected_pos)
        st.pyplot(fig_tl)
        plt.close(fig_tl)

        filter_mode = st.radio(
            "Filter Clickable Timestamps:",
            ["All Sampled Frames", "Suspicious / Uncertain Only"],
            horizontal=True,
            key="vid_tl_filter",
        )
        if filter_mode == "Suspicious / Uncertain Only":
            shown_frames = [r for r in timeline if r.get("prediction_outcome") in {OUTCOME_MANIPULATED, OUTCOME_UNCERTAIN}]
        else:
            shown_frames = timeline

        if shown_frames:
            cols_per_row = 6
            for start_i in range(0, len(shown_frames), cols_per_row):
                row_slice = shown_frames[start_i:start_i + cols_per_row]
                btn_cols = st.columns(cols_per_row)
                for c_idx, rec in enumerate(row_slice):
                    pos_i = int(rec["sample_position"])
                    emoji = _timeline_emoji(rec.get("timeline_color", "gray"))
                    prob_txt = f" {rec['fake_probability']:.2f}" if rec.get("fake_probability") is not None else " N/A"
                    label = f"{emoji} {rec['timestamp_label']}{prob_txt}"
                    btn_type = "primary" if pos_i == selected_pos else "secondary"
                    if btn_cols[c_idx].button(label, key=f"tl_btn_{pos_i}", type=btn_type, width="stretch"):
                        st.session_state["vid_selected_pos"] = pos_i
                        st.session_state["vid_frame_xai_result"] = None
                        st.session_state["vid_frame_xai_pos"] = None
                        st.rerun()
        else:
            st.success("✅ No manipulated or uncertain frames found under the selected filter.")

        # Selected Frame Detailed Inspection Panel
        sel_rec = timeline[selected_pos]
        st.markdown(
            f"#### 🖼️ Frame Inspection: `{sel_rec['timestamp_label']}` "
            f"(Frame `#{sel_rec['frame_index']}`, {sel_rec['timestamp_sec']:.3f}s)"
        )
        fc1, fc2, fc3 = st.columns([1.3, 1.0, 1.2])
        with fc1:
            if sel_rec.get("frame_preview") is not None:
                st.image(
                    sel_rec["frame_preview"],
                    caption=f"Decoded Video Frame @ {sel_rec['timestamp_label']} (Bounding box matches timeline color)",
                    width="stretch",
                )
        with fc2:
            if sel_rec.get("aligned_face") is not None:
                st.image(
                    sel_rec["aligned_face"],
                    caption="Primary Detected Face Crop (256x256)",
                    width="stretch",
                )
            else:
                st.warning("No verifiable human face crop extracted for this timestamp.")
        with fc3:
            emoji = _timeline_emoji(sel_rec.get("timeline_color", "gray"))
            st.markdown(f"**Frame Outcome:** {emoji} `{sel_rec.get('prediction_outcome', 'N/A')}`")
            if sel_rec.get("fake_probability") is not None:
                st.metric("Calibrated P(Manipulated)", f"{sel_rec['fake_probability']:.4f}")
                st.metric("Classifier Confidence", f"{sel_rec['classifier_confidence']*100:.2f}%")
                st.caption(
                    f"Uncalibrated P(Manipulated): `{sel_rec['uncalibrated_probability']:.4f}` | "
                    f"Faces detected in frame: `{sel_rec['detected_face_count']}`"
                )
            if sel_rec.get("uncertainty_reason"):
                st.warning(f"**Flag Reason:** {sel_rec['uncertainty_reason']}")

            face_preds = sel_rec.get("face_predictions", [])
            if len(face_preds) > 1:
                with st.expander(f"👥 Multiple Faces in Frame ({len(face_preds)})"):
                    for fp in face_preds:
                        st.write(
                            f"- Face #{fp['face_index'] + 1} (`{fp['bbox'][2]}x{fp['bbox'][3]}`): "
                            f"`{fp['prediction_outcome']}` (`P={fp['fake_probability']:.4f}`)"
                        )

            if sel_rec.get("aligned_face") is not None:
                if st.button("🔬 Run Full XAI (Grad-CAM + FFT + ELA) on This Frame", key=f"vid_xai_btn_{selected_pos}", width="stretch"):
                    with st.spinner("Generating Grad-CAM, 2D FFT & ELA maps for selected frame..."):
                        xai_res = pipeline.analyze_image(sel_rec["aligned_face"])
                        st.session_state["vid_frame_xai_result"] = xai_res
                        st.session_state["vid_frame_xai_pos"] = selected_pos

        if (
            st.session_state.get("vid_frame_xai_result") is not None
            and st.session_state.get("vid_frame_xai_pos") == selected_pos
        ):
            xai_res = st.session_state["vid_frame_xai_result"]
            st.markdown("##### 🔬 On-Demand Frame Explainability (Grad-CAM + 2D FFT + ELA)")
            xc1, xc2, xc3, xc4 = st.columns(4)
            with xc1:
                st.image(xai_res["gradcam_overlay"], caption="Grad-CAM Spatial Overlay", width="stretch")
            with xc2:
                st.image(xai_res.get("fft_log_colormap_img", xai_res["fft_spectrum_img"]), caption="2D FFT Log-Magnitude", width="stretch")
            with xc3:
                st.image(xai_res.get("fft_anomaly_map", xai_res["fft_spectrum_img"]), caption="FFT Anomaly Regions", width="stretch")
            with xc4:
                st.image(xai_res.get("ela_heatmap_color", xai_res["ela_image"]), caption=f"ELA Heatmap (Mean={xai_res['ela_mean_error']:.2f})", width="stretch")
            st.caption("Interpretive forensic visualizations only; Grad-CAM, FFT, and ELA do not constitute definitive standalone proof of manipulation.")

    # 4. Suspicious Time Intervals & Confidence Distribution
    st.markdown("---")
    c_susp, c_dist = st.columns([1.0, 1.4])
    with c_susp:
        st.subheader("⏱️ Suspicious Time Intervals")
        susp_intervals = results.get("suspicious_timestamps", [])
        if susp_intervals:
            df_susp = pd.DataFrame(susp_intervals)[
                ["start_timestamp", "end_timestamp", "duration_sec", "frame_count", "severity", "peak_fake_probability"]
            ]
            df_susp.columns = ["Start", "End", "Duration (s)", "Frames", "Severity", "Peak P(Fake)"]
            st.dataframe(df_susp, width="stretch", hide_index=True)
        else:
            st.success("✅ No suspicious or uncertain intervals detected across sampled timeline.")

        cd = results.get("confidence_distribution", {})
        if cd.get("count", 0) > 0:
            st.caption(
                f"Confidence Stats (`N={cd['count']}` face frames): "
                f"Mean=`{cd['conf_mean']*100:.1f}%`, Min=`{cd['conf_min']*100:.1f}%`, "
                f"Max=`{cd['conf_max']*100:.1f}%`, Prob Std=`{cd['prob_std']:.4f}`"
            )

    with c_dist:
        st.subheader("📊 Confidence & Outcome Distribution")
        fig_dist = plot_confidence_distribution(results)
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    # 5. Downloadable Structured Video Forensic Report (JSON, Markdown & PDF Certificate)
    st.markdown("---")
    st.subheader("📑 Export Video Forensic Report (Frame-Level)")
    report_dict = build_video_forensic_report(results)
    report_json_bytes = json.dumps(report_dict, indent=2).encode("utf-8")
    report_md_bytes = format_video_forensic_markdown(report_dict).encode("utf-8")
    base_stem = Path(results.get("filename", "video")).stem

    dl1, dl2, dl3 = st.columns(3)
    json_out_name = f"Video_Forensic_Report_{base_stem}.json"
    md_out_name = f"Video_Forensic_Report_{base_stem}.md"
    pdf_out_name = f"Video_Forensic_Certificate_{base_stem}.pdf"
    with dl1:
        st.download_button(
            label="📥 Download Video Report (JSON)",
            data=report_json_bytes,
            file_name=json_out_name,
            mime="application/json",
            on_click=_on_video_report_download,
            args=(db, results.get("case_id"), results.get("evidence_id"), "video_json", report_json_bytes, json_out_name),
            width="stretch",
        )
    with dl2:
        st.download_button(
            label="📄 Download Video Summary (Markdown)",
            data=report_md_bytes,
            file_name=md_out_name,
            mime="text/markdown",
            on_click=_on_video_report_download,
            args=(db, results.get("case_id"), results.get("evidence_id"), "video_markdown", report_md_bytes, md_out_name),
            width="stretch",
        )
    with dl3:
        if st.button("📑 Compile Video PDF Certificate", key="gen_vid_pdf_btn", width="stretch"):
            with st.spinner("Generating multi-page Video Forensic PDF Certificate..."):
                vid_events = []
                if db is not None and results.get("evidence_id"):
                    vid_events = [
                        ev
                        for ev in db.get_audit_events(evidence_id=results.get("evidence_id"))
                        if ev.get("case_id") in (results.get("case_id"), None)
                    ]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf_path = tmp_pdf.name
                try:
                    generate_video_pdf_report(results, output_path=tmp_pdf_path, audit_events=vid_events)
                    pdf_bytes = Path(tmp_pdf_path).read_bytes()
                finally:
                    if os.path.exists(tmp_pdf_path):
                        try:
                            os.unlink(tmp_pdf_path)
                        except OSError:
                            pass
                st.session_state["vid_pdf_bytes"] = pdf_bytes
                if db is not None:
                    db.record_report_generation(
                        case_id=results.get("case_id"),
                        evidence_id=results.get("evidence_id"),
                        report_format="video_pdf",
                        report_bytes=pdf_bytes,
                        output_name=pdf_out_name,
                        actor="streamlit_video_dashboard",
                    )

        if st.session_state.get("vid_pdf_bytes"):
            st.download_button(
                label="⬇️ Download Video Certificate (PDF)",
                data=st.session_state["vid_pdf_bytes"],
                file_name=pdf_out_name,
                mime="application/pdf",
                width="stretch",
                key="dl_vid_pdf_btn",
            )
