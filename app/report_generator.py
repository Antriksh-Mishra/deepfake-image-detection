"""
Forensic Report Generator (Image & Frame-Level Video PDF Certificates).

Generates publication-grade, multi-page ReportLab Forensic Examination Certificates
featuring:
  - Separate Case ID & Evidence ID references
  - Raw uploaded file SHA-256 digests distinct from processed crop digests
  - Validated 3-outcome calibrated policy (`AUTHENTIC`, `MANIPULATED`,
    `UNCERTAIN / NEEDS MANUAL REVIEW`) with decoupled advisory XAI signals
  - Safe multi-page pagination with wrapped cells, table `repeatRows=1`, and no clipping
  - Comprehensive frame-level Video Deepfake Certificates with timeline plots,
    suspicious intervals, and explicit `temporal_modeling=False` scope disclosure
"""

import hashlib
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from src.core.database import CHAIN_OF_CUSTODY_DISCLAIMER
except ImportError:
    CHAIN_OF_CUSTODY_DISCLAIMER = (
        "Software Audit Log vs. Legal Chain of Custody: Records captured by this workstation ledger "
        "constitute local software forensic audit logs and do NOT independently establish a legally "
        "admissible chain of custody without RFC 3161 TSA timestamping, PKI/HSM signatures, WORM storage, "
        "and attested human custody transfers."
    )


OUTCOME_HEX = {
    "green": "#10b981",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "gray": "#94a3b8",
}


def compute_sha256(pil_image: Image.Image) -> str:
    """Computes SHA-256 hash of encoded PIL image PNG bytes (post-crop reference)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def get_checkpoint_sha256() -> str:
    """Computes SHA-256 digest of the production model checkpoint."""
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_path = repo_root / "checkpoints" / "best_model.pth"
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "N/A"
    return "N/A"


def pil_to_flowable(
    pil_img: Image.Image,
    width: float = 1.45 * inch,
    height: float = 1.45 * inch,
) -> RLImage:
    """Converts a PIL image cleanly into a bounded ReportLab Image flowable."""
    rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)


def _aspect_fit_flowable(
    pil_img: Image.Image,
    max_width: float,
    max_height: float,
) -> RLImage:
    """Scales a PIL image preserving aspect ratio strictly within `(max_width, max_height)`."""
    w, h = max(pil_img.width, 1), max(pil_img.height, 1)
    scale = min(max_width / float(w), max_height / float(h))
    return pil_to_flowable(pil_img, width=max(12.0, w * scale), height=max(12.0, h * scale))


def _figure_to_flowable(fig: Figure, width: float, height: float) -> RLImage:
    """Exports an OOP Matplotlib `Figure` cleanly into a high-DPI ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)


def _xml_escape(text: Any) -> str:
    """Escapes characters that could break ReportLab Paragraph XML markup."""
    s = str(text if text is not None else "N/A")
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _resolve_verdict_theme(results: Dict[str, Any]) -> Tuple[colors.Color, str]:
    """
    Fixes B8: Checks uncertain / manual-review / no-face status BEFORE `is_fake`
    so borderline UNCERTAIN cases render in Amber rather than Red.
    """
    outcome = str(results.get("prediction_outcome") or "").upper()
    tier = str(results.get("verdict_tier") or results.get("overall_assessment") or "").upper()
    status = str(results.get("status") or "").lower()
    is_uncertain = bool(results.get("is_uncertain", False))
    is_fake = bool(results.get("is_fake", False))

    if (
        is_uncertain
        or "UNCERTAIN" in outcome
        or "INCONCLUSIVE" in tier
        or "UNCERTAIN" in tier
        or "INSUFFICIENT" in tier
        or "NO FACES" in tier
        or status == "no_faces_detected"
    ):
        return colors.HexColor("#d97706"), outcome or tier or "UNCERTAIN / NEEDS MANUAL REVIEW"
    if "MANIPULATED" in outcome or "MANIPULATED" in tier or "DEEPFAKE" in tier or is_fake:
        return colors.HexColor("#b91c1c"), outcome or tier or "MANIPULATED"
    return colors.HexColor("#047857"), outcome or tier or "AUTHENTIC"


def _add_page_chrome(canvas: Any, doc: Any) -> None:
    """Draws consistent running footer and upper rule on every page without clipping."""
    canvas.saveState()
    page_w, page_h = letter
    margin = 36

    if canvas.getPageNumber() > 1:
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.75)
        canvas.line(margin, page_h - 28, page_w - margin, page_h - 28)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(margin, page_h - 23, "DEEPFAKE FORENSIC SENTINEL — OFFICIAL EXAMINATION CERTIFICATE")

    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.75)
    canvas.line(margin, 32, page_w - margin, 32)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(
        margin,
        20,
        "Lead Developer: Antriksh Mishra | Dual-Domain Spatial & Spectral Workbench (Software Forensic Audit Report)",
    )
    canvas.drawRightString(page_w - margin, 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _build_radial_fft_flowable(active_rep: Dict[str, Any]) -> Optional[RLImage]:
    """Renders the azimuthal radial power spectrum comparison using OOP Matplotlib."""
    freqs = np.asarray(active_rep.get("azimuthal_frequencies", []))
    profile = np.asarray(active_rep.get("azimuthal_profile", []))
    if len(freqs) == 0 or len(profile) == 0:
        return None

    spec_details = active_rep.get("spectral_details", {})
    comp = active_rep.get("spectral_comparison", {})
    anomaly_bands = active_rep.get("fft_anomaly_bands", [])

    fig = Figure(figsize=(6.8, 2.7))
    ax = fig.add_subplot(111)

    for idx, (b_start, b_end) in enumerate(anomaly_bands):
        ax.axvspan(
            b_start,
            max(b_end, b_start + 1),
            color="#fca5a5",
            alpha=0.28,
            label="Anomalous Radial Band" if idx == 0 else None,
        )

    real_mean = comp.get("reference_real_mean")
    if real_mean is not None and len(real_mean) == len(freqs):
        ax.plot(freqs, real_mean, color="#059669", lw=1.6, label=f"Authentic Baseline (n={comp.get('num_authentic_refs', 0)})")
    fake_mean = comp.get("reference_fake_mean")
    if fake_mean is not None and len(fake_mean) == len(freqs):
        ax.plot(freqs, fake_mean, color="#d97706", lw=1.4, linestyle="-.", label=f"Suspicious Ref (n={comp.get('num_suspicious_refs', 0)})")

    ideal_decay = comp.get("ideal_decay")
    if ideal_decay is None or len(ideal_decay) != len(freqs):
        base_val = float(profile[1]) if len(profile) > 1 else 1.0
        intercept = base_val + 1.69 * float(np.log1p(1.0))
        ideal_decay = np.maximum(0.0, intercept - 1.69 * np.log1p(np.maximum(freqs, 0.0)))
    ax.plot(freqs, ideal_decay, color="#94a3b8", linestyle="--", lw=1.2, label="Optical 1/f Log Decay")
    ax.plot(freqs, profile, color="#1d4ed8", lw=2.0, label="Sample Profile")

    peak_idxs = [p for p in spec_details.get("peak_indices", []) if 0 <= p < len(freqs)]
    if peak_idxs:
        ax.scatter(
            [freqs[p] for p in peak_idxs],
            [profile[p] for p in peak_idxs],
            color="#dc2626",
            s=36,
            zorder=5,
            label="Periodic Spikes",
        )

    ax.set_xlabel("Spatial Frequency Radius (cycles/image)", fontsize=8)
    ax.set_ylabel("Mean Log Magnitude", fontsize=8)
    ax.set_title("2D FFT Azimuthal Radial Power Comparison", fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    return _figure_to_flowable(fig, width=6.9 * inch, height=2.65 * inch)


def _build_video_timeline_flowable(results: Dict[str, Any]) -> Optional[RLImage]:
    """Renders calibrated frame-level timeline chart cleanly without pyplot global state."""
    timeline = results.get("timeline", [])
    if not timeline:
        return None

    cal_meta = results.get("calibration_metadata", {})
    tau = float(cal_meta.get("decision_threshold", 0.50))
    u_low = float(cal_meta.get("uncertain_low", 0.40))
    u_high = float(cal_meta.get("uncertain_high", 0.60))

    fig = Figure(figsize=(7.2, 2.7))
    ax = fig.add_subplot(111)
    ax.axhspan(u_low, u_high, color="#fde68a", alpha=0.35, label=f"Uncertainty Band [{u_low:.3f}, {u_high:.3f}]")
    ax.axhline(tau, color="#475569", linestyle="--", linewidth=1.1, label=f"Threshold tau = {tau:.2f}")

    face_pts = [r for r in timeline if r.get("has_face") and r.get("fake_probability") is not None]
    if face_pts:
        ax.plot(
            [r["timestamp_sec"] for r in face_pts],
            [r["fake_probability"] for r in face_pts],
            color="#94a3b8",
            linewidth=1.1,
            alpha=0.75,
        )

    color_groups = [
        ("green", OUTCOME_HEX["green"], "o", "Authentic"),
        ("red", OUTCOME_HEX["red"], "o", "Manipulated"),
        ("yellow", OUTCOME_HEX["yellow"], "D", "Uncertain"),
        ("gray", OUTCOME_HEX["gray"], "x", "No Face"),
    ]
    for c_name, c_hex, marker, lbl in color_groups:
        grp = [r for r in timeline if r.get("timeline_color") == c_name]
        if not grp:
            continue
        xs = [r["timestamp_sec"] for r in grp]
        ys = [r["fake_probability"] if r.get("fake_probability") is not None else 0.03 for r in grp]
        ax.scatter(xs, ys, c=c_hex, marker=marker, s=42, edgecolors="#0f172a" if marker != "x" else None, linewidths=0.6, label=lbl, zorder=4)

    ax.set_ylim(-0.04, 1.14)
    ax.set_xlabel("Timestamp (seconds)", fontsize=8)
    ax.set_ylabel("Calibrated P(Manipulated)", fontsize=8)
    ax.set_title("Frame-Level Video Deepfake Timeline (Independent Per-Frame Inference)", fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc="upper center", ncol=3, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    return _figure_to_flowable(fig, width=7.0 * inch, height=2.55 * inch)


def _get_pdf_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Normal": base["Normal"],
        "Title": ParagraphStyle(
            "DFSTitle",
            parent=base["Heading1"],
            fontSize=16.5,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            spaceAfter=2,
        ),
        "SubTitle": ParagraphStyle(
            "DFSSubTitle",
            parent=base["Normal"],
            fontSize=9.0,
            leading=12.5,
            textColor=colors.HexColor("#475569"),
        ),
        "Section": ParagraphStyle(
            "DFSSection",
            parent=base["Heading3"],
            fontSize=10.2,
            leading=13.5,
            textColor=colors.HexColor("#1e3a8a"),
            fontName="Helvetica-Bold",
            spaceBefore=6,
            spaceAfter=4,
        ),
        "Cell": ParagraphStyle(
            "DFSCell",
            parent=base["Normal"],
            fontSize=8.0,
            leading=10.5,
            textColor=colors.HexColor("#1e293b"),
        ),
        "CellSmall": ParagraphStyle(
            "DFSCellSmall",
            parent=base["Normal"],
            fontSize=7.2,
            leading=9.5,
            textColor=colors.HexColor("#334155"),
        ),
        "Mono": ParagraphStyle(
            "DFSMono",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=7.0,
            leading=9.2,
            textColor=colors.HexColor("#0f172a"),
        ),
    }


def _append_audit_and_custody_section(
    story: List[Any],
    st_map: Dict[str, ParagraphStyle],
    audit_events: Optional[Sequence[Dict[str, Any]]],
    is_video: bool,
) -> None:
    """Appends immutable `audit_events` summary (if any), SWGDE/ENFSI disclaimer, and signature stamp."""
    if audit_events:
        story.append(Paragraph("IMMUTABLE AUDIT LIFECYCLE EVENTS (`audit_events`)", st_map["Section"]))
        ev_rows = [
            [
                Paragraph("<b>Event ID</b>", st_map["CellSmall"]),
                Paragraph("<b>Lifecycle Type</b>", st_map["CellSmall"]),
                Paragraph("<b>UTC Timestamp</b>", st_map["CellSmall"]),
                Paragraph("<b>SHA-256 Digest</b>", st_map["CellSmall"]),
                Paragraph("<b>Actor</b>", st_map["CellSmall"]),
            ]
        ]
        for ev in list(audit_events)[:12]:
            sha_v = str(ev.get("sha256_at_event", "N/A"))
            sha_short = f"{sha_v[:20]}...{sha_v[-8:]}" if len(sha_v) > 32 else sha_v
            ev_rows.append(
                [
                    Paragraph(_xml_escape(ev.get("event_id", "")), st_map["Mono"]),
                    Paragraph(_xml_escape(ev.get("event_type", "")), st_map["CellSmall"]),
                    Paragraph(_xml_escape(ev.get("event_timestamp_utc", "")), st_map["CellSmall"]),
                    Paragraph(_xml_escape(sha_short), st_map["Mono"]),
                    Paragraph(_xml_escape(ev.get("actor", "")), st_map["CellSmall"]),
                ]
            )
        ev_tbl = Table(
            ev_rows,
            colWidths=[1.55 * inch, 1.05 * inch, 1.45 * inch, 1.95 * inch, 1.1 * inch],
            repeatRows=1,
        )
        ev_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 3),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(ev_tbl)
        story.append(Spacer(1, 8))

    # Scientific Limitations & Legal Chain-of-Custody Distinction
    limitations_paragraphs = [
        (
            "Under guidelines from SWGDE and ENFSI, neural classification scores and XAI visualizations "
            "(Grad-CAM heatmaps, 2D FFT spectra, Error Level Analysis) provide statistical corroboration "
            "and interpretive diagnostic clues rather than standalone proof of forgery or authenticity."
        ),
        CHAIN_OF_CUSTODY_DISCLAIMER,
    ]
    if is_video:
        limitations_paragraphs.insert(
            0,
            "Video Scope Caveat: Sampled frames are classified independently via DualDomainSentinelNet "
            "(analysis_scope='frame_level_inference', temporal_modeling=False). No 3D temporal CNN, "
            "optical-flow, or inter-frame sequence model is active.",
        )

    story.append(
        KeepTogether(
            [
                Paragraph("SCIENTIFIC LIMITATIONS, AUDIT INTEGRITY & LEGAL DISCLAIMER", st_map["Section"]),
                Paragraph(
                    "<br/><br/>".join(_xml_escape(p) for p in limitations_paragraphs),
                    st_map["CellSmall"],
                ),
                Spacer(1, 10),
                Table(
                    [
                        [
                            Paragraph(
                                "<b>Lead Researcher &amp; Capstone Developer:</b><br/>"
                                "<b>Antriksh Mishra</b><br/>"
                                "Department of Computer Science &amp; Engineering<br/>"
                                "Deepfake Forensic Sentinel (B.Tech Final Year Project)",
                                st_map["CellSmall"],
                            ),
                            Paragraph(
                                "<b>Forensic Engine &amp; Calibration Metadata:</b><br/>"
                                "DualDomainSentinelNet (EfficientNet-B0 + 3-SRM + Cross-Attention)<br/>"
                                "Trained with LabelSmoothingBCELoss(0.06); Validated Temperature Scaling<br/>"
                                "Software Forensic Audit Digest Attached",
                                st_map["CellSmall"],
                            ),
                        ]
                    ],
                    colWidths=[3.55 * inch, 3.55 * inch],
                    style=TableStyle(
                        [
                            ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.HexColor("#94a3b8")),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                            ("PADDING", (0, 0), (-1, -1), 6),
                        ]
                    ),
                ),
            ]
        )
    )


def _build_image_pdf_story(
    analysis_results: Dict[str, Any],
    st_map: Dict[str, ParagraphStyle],
    audit_events: Optional[Sequence[Dict[str, Any]]],
) -> List[Any]:
    story: List[Any] = []

    story.append(Paragraph("DEEPFAKE FORENSIC EXAMINATION CERTIFICATE (IMAGE)", st_map["Title"]))
    story.append(
        Paragraph(
            "Dual-Domain Spatial &amp; 2D FFT Spectral Verification &bull; Validated Calibration &amp; Explainable AI",
            st_map["SubTitle"],
        )
    )
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=8))

    aligned_face = analysis_results.get("aligned_face")
    crop_sha = compute_sha256(aligned_face) if isinstance(aligned_face, Image.Image) else "N/A"
    raw_sha = analysis_results.get("raw_file_sha256") or "N/A (Raw bytes not supplied)"
    ckpt_hash = get_checkpoint_sha256()
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    case_ref = analysis_results.get("case_id") or f"DFS-{int(time.time())}"
    evd_ref = analysis_results.get("evidence_id") or "N/A"
    file_sz = analysis_results.get("file_size_bytes")
    dims_str = analysis_results.get("media_dimensions") or (
        f"{aligned_face.width}x{aligned_face.height} (crop)" if isinstance(aligned_face, Image.Image) else "unknown"
    )
    size_disp = f"{dims_str} ({int(file_sz):,} bytes)" if file_sz else str(dims_str)

    cal_meta = analysis_results.get("calibration_metadata", {}) or {}
    cal_summary = (
        f"T={float(cal_meta.get('temperature', 1.0)):.4f} | "
        f"tau={float(cal_meta.get('decision_threshold', 0.50)):.2f} | "
        f"Uncertain Band=[{float(cal_meta.get('uncertain_low', 0.40)):.3f}, "
        f"{float(cal_meta.get('uncertain_high', 0.60)):.3f}]"
    )

    has_face = bool(analysis_results.get("has_face", analysis_results.get("face_detected", True)))
    img_type = str(analysis_results.get("image_type", "Human Face Portrait" if has_face else "General Object"))
    if has_face:
        loc_desc = f"Haar Face Alignment (256x256 Lanczos) | Detected Faces: {int(analysis_results.get('faces_count', 1))}"
    else:
        loc_desc = f"Media Type: {_xml_escape(img_type)} | Object-Aware Localization (256x256 Lanczos)"

    meta_data = [
        [Paragraph("<b>Lead Examiner:</b>", st_map["Cell"]), Paragraph("<b>Antriksh Mishra</b> (B.Tech CSE Capstone)", st_map["Cell"])],
        [Paragraph("<b>Case / Evidence ID:</b>", st_map["Cell"]), Paragraph(f"<b>{_xml_escape(case_ref)}</b> &nbsp;|&nbsp; Evidence: <code>{_xml_escape(evd_ref)}</code>", st_map["Cell"])],
        [Paragraph("<b>Examination UTC:</b>", st_map["Cell"]), Paragraph(_xml_escape(timestamp_str), st_map["Cell"])],
        [Paragraph("<b>Dimensions / Size:</b>", st_map["Cell"]), Paragraph(_xml_escape(size_disp), st_map["Cell"])],
        [Paragraph("<b>Calibration Policy:</b>", st_map["Cell"]), Paragraph(_xml_escape(cal_summary), st_map["Cell"])],
        [Paragraph("<b>Raw Uploaded SHA-256:</b>", st_map["Cell"]), Paragraph(_xml_escape(raw_sha), st_map["Mono"])],
        [Paragraph("<b>Aligned Crop SHA-256:</b>", st_map["Cell"]), Paragraph(_xml_escape(crop_sha), st_map["Mono"])],
        [Paragraph("<b>Checkpoint SHA-256:</b>", st_map["Cell"]), Paragraph(_xml_escape(ckpt_hash), st_map["Mono"])],
        [
            Paragraph("<b>Localization &amp; Media:</b>", st_map["Cell"]),
            Paragraph(loc_desc, st_map["Cell"]),
        ],
    ]
    meta_tbl = Table(meta_data, colWidths=[1.75 * inch, 5.35 * inch])
    meta_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 3.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 8))

    # 2. Calibrated 3-Outcome Verdict Box (B8 fixed)
    verdict_color, resolved_outcome = _resolve_verdict_theme(analysis_results)
    verdict_badge = analysis_results.get("verdict_badge") or f"FORENSIC OUTCOME: {resolved_outcome}"
    verdict_explanation = analysis_results.get("verdict_explanation") or "Dual-domain calibrated examination completed."
    confidence_val = float(analysis_results.get("confidence_percent", 0.0))
    fake_prob_val = float(analysis_results.get("fake_probability", 0.5))
    latency_val = float(analysis_results.get("inference_time_ms", 0.0))

    verdict_tbl = Table(
        [
            [Paragraph(f"<font color='white' size=11.5><b>{_xml_escape(verdict_badge)}</b></font>", st_map["Normal"])],
            [
                Paragraph(
                    f"<font color='white' size=9>Outcome: <b>{_xml_escape(resolved_outcome)}</b> &nbsp;|&nbsp; "
                    f"Calibrated Confidence: <b>{confidence_val:.2f}%</b> &nbsp;|&nbsp; "
                    f"Calibrated P(Manipulated): <b>{fake_prob_val:.4f}</b> &nbsp;|&nbsp; "
                    f"Latency: <b>{latency_val:.1f} ms</b></font>",
                    st_map["Normal"],
                )
            ],
            [Paragraph(f"<font color='#f8fafc' size=8><i>{_xml_escape(verdict_explanation)}</i></font>", st_map["Normal"])],
        ],
        colWidths=[7.1 * inch],
    )
    verdict_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(verdict_tbl)
    story.append(Spacer(1, 8))

    # 3. Decoupled Primary Classifier vs Advisory XAI Evidence Table (NO hardcoded 0.50N+0.30S+0.20E!)
    ev = analysis_results.get("forensic_evidence", {}) or {}
    p_n = float(ev.get("neural_fake_prob", fake_prob_val)) * 100.0
    p_uncal = float(ev.get("neural_uncalibrated_prob", ev.get("neural_fake_prob", fake_prob_val))) * 100.0
    p_s = float(ev.get("spectral_anomaly_prob", 0.0)) * 100.0
    p_e = float(ev.get("ela_anomaly_prob", 0.0)) * 100.0
    spec_det = analysis_results.get("spectral_details", {}) or {}
    ela_det = analysis_results.get("ela_details", {}) or {}

    score_rows = [
        [
            Paragraph("<b>Forensic Stream</b>", st_map["Cell"]),
            Paragraph("<b>Methodology &amp; Policy Role</b>", st_map["Cell"]),
            Paragraph("<b>Measured Signal</b>", st_map["Cell"]),
            Paragraph("<b>Diagnostic Notes</b>", st_map["Cell"]),
        ],
        [
            Paragraph("<b>Primary Neural Classifier</b>", st_map["Cell"]),
            Paragraph("DualDomainSentinelNet + Temp Scaling<br/><i>(Authoritative Classifier Output)</i>", st_map["CellSmall"]),
            Paragraph(f"<b>Calibrated: {p_n:.1f}%</b><br/>Uncal: {p_uncal:.1f}%", st_map["CellSmall"]),
            Paragraph(f"Outcome: <b>{_xml_escape(resolved_outcome)}</b> (tau={float(cal_meta.get('decision_threshold', 0.50)):.2f})", st_map["CellSmall"]),
        ],
        [
            Paragraph("<b>Auxiliary 2D FFT Spectrum</b>", st_map["Cell"]),
            Paragraph("Azimuthal Log-Power vs 1/f Optical Decay<br/><i>(Advisory Only — Zero Override)</i>", st_map["CellSmall"]),
            Paragraph(f"<b>{p_s:.1f}%</b> anomaly index", st_map["CellSmall"]),
            Paragraph(
                f"Peaks: {len(spec_det.get('detected_peaks', []))} | Spikes Flag: {'Yes' if spec_det.get('has_deconv_spikes') else 'No'}",
                st_map["CellSmall"],
            ),
        ],
        [
            Paragraph("<b>Auxiliary Quantization (ELA)</b>", st_map["Cell"]),
            Paragraph("JPEG Recompression Residuals (Q=90)<br/><i>(Advisory Only — Zero Override)</i>", st_map["CellSmall"]),
            Paragraph(f"<b>{p_e:.1f}%</b> mismatch index", st_map["CellSmall"]),
            Paragraph(
                f"Mean ELA: {float(analysis_results.get('ela_mean_error', 0.0)):.2f} | Quad Ratio: {float(ela_det.get('quad_discrepancy_ratio', 0.0)):.3f}",
                st_map["CellSmall"],
            ),
        ],
    ]
    score_tbl = Table(score_rows, colWidths=[1.65 * inch, 2.25 * inch, 1.25 * inch, 1.95 * inch], repeatRows=1)
    score_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eff6ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(score_tbl)
    story.append(Spacer(1, 8))

    # 4. Multi-Modal Forensic Visual Evidence Artifacts Grid
    fft_img = analysis_results.get("fft_log_colormap_img") or analysis_results.get("fft_spectrum_img") or aligned_face
    ela_img = analysis_results.get("ela_heatmap_color") or analysis_results.get("ela_image") or aligned_face
    grad_img = analysis_results.get("gradcam_overlay") or aligned_face

    if isinstance(aligned_face, Image.Image):
        img_sz = 1.52 * inch
        cap_a = "<b>(A) Primary Face</b><br/>256x256 Aligned Crop" if has_face else f"<b>(A) Localized Media</b><br/>{_xml_escape(img_type[:22])}"
        ev_grid = Table(
            [
                [
                    _aspect_fit_flowable(aligned_face, img_sz, img_sz),
                    _aspect_fit_flowable(grad_img, img_sz, img_sz),
                    _aspect_fit_flowable(fft_img, img_sz, img_sz),
                    _aspect_fit_flowable(ela_img, img_sz, img_sz),
                ],
                [
                    Paragraph(cap_a, st_map["CellSmall"]),
                    Paragraph("<b>(B) Spatial Grad-CAM</b><br/>Neural Focus Overlay", st_map["CellSmall"]),
                    Paragraph("<b>(C) 2D FFT Log-Mag</b><br/>Shifted Spectral Map", st_map["CellSmall"]),
                    Paragraph("<b>(D) ELA Residuals</b><br/>Q=90 Heatmap", st_map["CellSmall"]),
                ],
            ],
            colWidths=[1.775 * inch] * 4,
        )
        ev_grid.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(
            KeepTogether(
                [
                    Paragraph("MULTI-MODAL FORENSIC EVIDENCE ARTIFACTS", st_map["Section"]),
                    ev_grid,
                    Spacer(1, 8),
                ]
            )
        )

    # Radial Power Spectrum Comparison Chart (if computed)
    radial_flowable = _build_radial_fft_flowable(analysis_results)
    if radial_flowable is not None:
        story.append(KeepTogether([radial_flowable, Spacer(1, 8)]))

    # Multi-Face Per-Subject Breakdown Table (if >1 face detected)
    multi_faces = analysis_results.get("multi_face_reports", [])
    if len(multi_faces) > 1:
        mf_rows = [
            [
                Paragraph("<b>Subject</b>", st_map["CellSmall"]),
                Paragraph("<b>Bounding Box (x,y,w,h)</b>", st_map["CellSmall"]),
                Paragraph("<b>Outcome</b>", st_map["CellSmall"]),
                Paragraph("<b>P(Manip)</b>", st_map["CellSmall"]),
                Paragraph("<b>Conf (%)</b>", st_map["CellSmall"]),
                Paragraph("<b>Dominant Grad-CAM Region</b>", st_map["CellSmall"]),
            ]
        ]
        for rep in multi_faces:
            dom = (rep.get("spatial_landmarks") or {}).get("dominant_region", "N/A")
            mf_rows.append(
                [
                    Paragraph(f"Face #{int(rep.get('face_index', 0)) + 1}{' (Primary)' if rep.get('is_primary') else ''}", st_map["CellSmall"]),
                    Paragraph(_xml_escape(str(rep.get("bbox"))), st_map["Mono"]),
                    Paragraph(_xml_escape(rep.get("prediction_outcome", "N/A")), st_map["CellSmall"]),
                    Paragraph(f"{float(rep.get('fake_probability', 0.0)):.4f}", st_map["CellSmall"]),
                    Paragraph(f"{float(rep.get('confidence_percent', 0.0)):.1f}%", st_map["CellSmall"]),
                    Paragraph(_xml_escape(dom), st_map["CellSmall"]),
                ]
            )
        mf_tbl = Table(
            mf_rows,
            colWidths=[1.05 * inch, 1.3 * inch, 1.45 * inch, 0.75 * inch, 0.75 * inch, 1.8 * inch],
            repeatRows=1,
        )
        mf_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 3.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(Paragraph(f"MULTI-FACE PER-SUBJECT SUMMARY ({len(multi_faces)} DETECTED FACES)", st_map["Section"]))
        story.append(mf_tbl)
        story.append(Spacer(1, 8))

    # Forensic Observations (Interpretive)
    obs = analysis_results.get("forensic_observations", [])
    if obs:
        story.append(Paragraph("INTERPRETIVE FORENSIC OBSERVATIONS", st_map["Section"]))
        for item in obs:
            story.append(Paragraph(f"&bull; {_xml_escape(item)}", st_map["CellSmall"]))
        story.append(Spacer(1, 6))

    _append_audit_and_custody_section(story, st_map, audit_events, is_video=False)
    return story


def _build_video_pdf_story(
    video_results: Dict[str, Any],
    st_map: Dict[str, ParagraphStyle],
    audit_events: Optional[Sequence[Dict[str, Any]]],
) -> List[Any]:
    story: List[Any] = []

    story.append(Paragraph("DEEPFAKE FORENSIC EXAMINATION CERTIFICATE (VIDEO)", st_map["Title"]))
    story.append(
        Paragraph(
            "Frame-Level Dual-Domain Spatial &amp; 2D FFT Spectral Analysis &bull; "
            "analysis_scope='frame_level_inference' (temporal_modeling=False)",
            st_map["SubTitle"],
        )
    )
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=8))

    f_meta = video_results.get("file_metadata", {}) or {}
    raw_sha = video_results.get("sha256") or video_results.get("raw_file_sha256") or "N/A"
    ckpt_hash = get_checkpoint_sha256()
    case_ref = video_results.get("case_id") or f"DFS-{int(time.time())}"
    evd_ref = video_results.get("evidence_id") or "N/A"
    cal_meta = video_results.get("calibration_metadata", {}) or {}
    samp_cfg = video_results.get("sampling_config", {}) or {}
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    meta_rows = [
        [Paragraph("<b>Lead Examiner:</b>", st_map["Cell"]), Paragraph("<b>Antriksh Mishra</b> (B.Tech CSE Capstone)", st_map["Cell"])],
        [Paragraph("<b>Case / Evidence ID:</b>", st_map["Cell"]), Paragraph(f"<b>{_xml_escape(case_ref)}</b> &nbsp;|&nbsp; Evidence: <code>{_xml_escape(evd_ref)}</code>", st_map["Cell"])],
        [Paragraph("<b>Filename &amp; UTC Time:</b>", st_map["Cell"]), Paragraph(f"{_xml_escape(video_results.get('filename', 'video'))} &nbsp;|&nbsp; {timestamp_str}", st_map["Cell"])],
        [
            Paragraph("<b>Container Geometry:</b>", st_map["Cell"]),
            Paragraph(
                f"{int(f_meta.get('width', 0))}x{int(f_meta.get('height', 0))} @ {float(video_results.get('fps', 0.0)):.2f} FPS &nbsp;|&nbsp; "
                f"Duration: {video_results.get('duration_formatted', '00:00.00')} ({float(video_results.get('duration_sec', 0.0)):.2f}s) &nbsp;|&nbsp; "
                f"Size: {float(f_meta.get('file_size_mb', 0.0)):.2f} MB",
                st_map["Cell"],
            ),
        ],
        [
            Paragraph("<b>Sampling Configuration:</b>", st_map["Cell"]),
            Paragraph(
                f"Mode={_xml_escape(samp_cfg.get('sample_mode', 'fps'))}, Sampled={int(video_results.get('processed_frames', 0))}/{int(video_results.get('frame_count', 0))} frames, "
                f"Faces={int(video_results.get('detected_faces', 0))} ({int(video_results.get('frames_with_faces', 0))} frames w/ face)",
                st_map["Cell"],
            ),
        ],
        [
            Paragraph("<b>Calibration Policy:</b>", st_map["Cell"]),
            Paragraph(
                f"T={float(cal_meta.get('temperature', 1.0)):.4f} | tau={float(cal_meta.get('decision_threshold', 0.50)):.2f} | "
                f"Band=[{float(cal_meta.get('uncertain_low', 0.40)):.3f}, {float(cal_meta.get('uncertain_high', 0.60)):.3f}]",
                st_map["Cell"],
            ),
        ],
        [Paragraph("<b>Raw Video SHA-256:</b>", st_map["Cell"]), Paragraph(_xml_escape(raw_sha), st_map["Mono"])],
        [Paragraph("<b>Checkpoint SHA-256:</b>", st_map["Cell"]), Paragraph(_xml_escape(ckpt_hash), st_map["Mono"])],
    ]
    meta_tbl = Table(meta_rows, colWidths=[1.75 * inch, 5.35 * inch])
    meta_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 8))

    # Overall Assessment Box
    verdict_color, resolved_outcome = _resolve_verdict_theme(video_results)
    verdict_badge = video_results.get("verdict_badge") or f"VIDEO ASSESSMENT: {video_results.get('overall_assessment', resolved_outcome)}"
    agg_prob = video_results.get("aggregated_fake_probability")
    agg_prob_s = f"{float(agg_prob):.4f}" if agg_prob is not None else "N/A"
    agg_conf = float(video_results.get("aggregated_confidence_percent", 0.0))

    verdict_tbl = Table(
        [
            [Paragraph(f"<font color='white' size=11.5><b>{_xml_escape(verdict_badge)}</b></font>", st_map["Normal"])],
            [
                Paragraph(
                    f"<font color='white' size=9>Overall Outcome: <b>{_xml_escape(video_results.get('overall_assessment', resolved_outcome))}</b> &nbsp;|&nbsp; "
                    f"Trimmed Mean P(Manipulated): <b>{agg_prob_s}</b> &nbsp;|&nbsp; "
                    f"Confidence: <b>{agg_conf:.2f}%</b> &nbsp;|&nbsp; "
                    f"Latency: <b>{float(video_results.get('inference_time_ms', 0.0)):.1f} ms</b></font>",
                    st_map["Normal"],
                )
            ],
            [Paragraph(f"<font color='#f8fafc' size=8><i>{_xml_escape(video_results.get('verdict_explanation', ''))}</i></font>", st_map["Normal"])],
        ],
        colWidths=[7.1 * inch],
    )
    verdict_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(verdict_tbl)
    story.append(Spacer(1, 8))

    # Frame Statistics Summary Table
    stat_rows = [
        [
            Paragraph("<b>Authentic Frames</b>", st_map["CellSmall"]),
            Paragraph("<b>Manipulated Frames</b>", st_map["CellSmall"]),
            Paragraph("<b>Uncertain Frames</b>", st_map["CellSmall"]),
            Paragraph("<b>No-Face Frames</b>", st_map["CellSmall"]),
            Paragraph("<b>Median / P90 P(Fake)</b>", st_map["CellSmall"]),
            Paragraph("<b>Quorum Safeguard</b>", st_map["CellSmall"]),
        ],
        [
            Paragraph(f"{int(video_results.get('authentic_frame_count', 0))} ({float(video_results.get('authentic_frame_pct', 0.0)):.1f}%)", st_map["CellSmall"]),
            Paragraph(f"{int(video_results.get('manipulated_frame_count', 0))} ({float(video_results.get('manipulated_frame_pct', 0.0)):.1f}%)", st_map["CellSmall"]),
            Paragraph(f"{int(video_results.get('uncertain_frame_count', 0))} ({float(video_results.get('uncertain_frame_pct', 0.0)):.1f}%)", st_map["CellSmall"]),
            Paragraph(f"{int(video_results.get('frames_without_faces', 0))}", st_map["CellSmall"]),
            Paragraph(
                f"{float(video_results.get('median_fake_probability') or 0.0):.3f} / {float(video_results.get('p90_fake_probability') or 0.0):.3f}",
                st_map["CellSmall"],
            ),
            Paragraph(
                f"Min {int(video_results.get('sustained_quorum_required', 2))} frames"
                f"{' (1 spike suppressed)' if video_results.get('isolated_spike_suppressed') else ''}",
                st_map["CellSmall"],
            ),
        ],
    ]
    stat_tbl = Table(stat_rows, colWidths=[1.15 * inch, 1.2 * inch, 1.1 * inch, 0.9 * inch, 1.25 * inch, 1.5 * inch])
    stat_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    story.append(stat_tbl)
    story.append(Spacer(1, 8))

    # Timeline Chart
    tl_chart = _build_video_timeline_flowable(video_results)
    if tl_chart is not None:
        story.append(KeepTogether([tl_chart, Spacer(1, 8)]))

    # Suspicious Intervals Table
    susp = video_results.get("suspicious_timestamps", [])
    story.append(Paragraph("FLAGGED SUSPICIOUS / UNCERTAIN TIMESTAMP INTERVALS", st_map["Section"]))
    if susp:
        susp_rows = [
            [
                Paragraph("<b>Start</b>", st_map["CellSmall"]),
                Paragraph("<b>End</b>", st_map["CellSmall"]),
                Paragraph("<b>Duration (s)</b>", st_map["CellSmall"]),
                Paragraph("<b>Frames</b>", st_map["CellSmall"]),
                Paragraph("<b>Severity</b>", st_map["CellSmall"]),
                Paragraph("<b>Peak P(Fake)</b>", st_map["CellSmall"]),
            ]
        ]
        for iv in susp:
            susp_rows.append(
                [
                    Paragraph(_xml_escape(iv.get("start_timestamp")), st_map["Mono"]),
                    Paragraph(_xml_escape(iv.get("end_timestamp")), st_map["Mono"]),
                    Paragraph(f"{float(iv.get('duration_sec', 0.0)):.2f}", st_map["CellSmall"]),
                    Paragraph(f"{int(iv.get('frame_count', 0))}", st_map["CellSmall"]),
                    Paragraph(_xml_escape(iv.get("severity")), st_map["CellSmall"]),
                    Paragraph(f"{float(iv.get('peak_fake_probability', 0.0)):.4f}", st_map["CellSmall"]),
                ]
            )
        susp_tbl = Table(susp_rows, colWidths=[1.0 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch, 2.2 * inch, 1.1 * inch], repeatRows=1)
        susp_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 3.5),
                ]
            )
        )
        story.append(susp_tbl)
    else:
        story.append(Paragraph("No sustained suspicious or uncertain timestamp intervals were flagged across sampled frames.", st_map["CellSmall"]))
    story.append(Spacer(1, 8))

    # Sampled Timeline Table (paginated safely with repeatRows=1)
    timeline = video_results.get("timeline", [])
    if timeline:
        story.append(Paragraph("SAMPLED FRAME-BY-FRAME TIMELINE TELEMETRY", st_map["Section"]))
        tl_rows = [
            [
                Paragraph("<b>Frame #</b>", st_map["CellSmall"]),
                Paragraph("<b>Timestamp</b>", st_map["CellSmall"]),
                Paragraph("<b>Faces</b>", st_map["CellSmall"]),
                Paragraph("<b>Outcome</b>", st_map["CellSmall"]),
                Paragraph("<b>Calibrated P(Fake)</b>", st_map["CellSmall"]),
            ]
        ]
        for rec in timeline[:60]:
            fp = rec.get("fake_probability")
            tl_rows.append(
                [
                    Paragraph(str(rec.get("frame_index", 0)), st_map["Mono"]),
                    Paragraph(_xml_escape(rec.get("timestamp_label", "00:00.00")), st_map["Mono"]),
                    Paragraph(str(rec.get("detected_face_count", 0)), st_map["CellSmall"]),
                    Paragraph(_xml_escape(rec.get("prediction_outcome", "N/A")), st_map["CellSmall"]),
                    Paragraph(f"{float(fp):.4f}" if fp is not None else "N/A", st_map["CellSmall"]),
                ]
            )
        tl_tbl = Table(tl_rows, colWidths=[0.9 * inch, 1.2 * inch, 0.8 * inch, 2.8 * inch, 1.4 * inch], repeatRows=1)
        tl_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(tl_tbl)
        story.append(Spacer(1, 8))

    _append_audit_and_custody_section(story, st_map, audit_events, is_video=True)
    return story


def generate_pdf_report(
    analysis_results: Dict[str, Any],
    output_path: str = "forensic_report.pdf",
    audit_events: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """
    Builds and writes a publication-grade multi-page PDF Forensic Examination
    Certificate for either Image or Frame-Level Video examination results.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=44,
    )
    st_map = _get_pdf_styles()
    is_video = any(k in analysis_results for k in ("timeline", "overall_assessment", "duration_sec"))
    if is_video:
        story = _build_video_pdf_story(analysis_results, st_map, audit_events=audit_events)
    else:
        story = _build_image_pdf_story(analysis_results, st_map, audit_events=audit_events)

    doc.build(story, onFirstPage=_add_page_chrome, onLaterPages=_add_page_chrome)
    return str(out_file)


def generate_video_pdf_report(
    video_results: Dict[str, Any],
    output_path: str = "video_forensic_report.pdf",
    audit_events: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Explicit alias generating a frame-level Video Forensic PDF Certificate."""
    return generate_pdf_report(video_results, output_path=output_path, audit_events=audit_events)
