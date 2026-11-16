"""
Explainable AI (XAI) Visualization & Unified Forensic Evidence Panel Module.
Provides publication-grade Matplotlib plots and Streamlit UI components for:
  - Unified 8-Item Forensic Evidence Panel
  - Per-Face Spatial Grad-CAM Overlays & Landmark Heatmaps (Multi-Face Ready)
  - 2D FFT Magnitude, Log-Magnitude Colormap, Anomaly Regions & Real-vs-Suspicious Radial Comparison
  - Error Level Analysis (ELA) Multi-Quality Recompression Comparisons & Explicit Limitations
"""

from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def plot_radial_spectrum_comparison(rep: Dict[str, Any]) -> plt.Figure:
    """
    Plots azimuthal radial log-power profile of the query face against theoretical 1/f decay,
    empirical local authentic vs suspicious reference spectra, anomalous radial bands, and spikes.
    """
    freqs = np.asarray(rep.get("azimuthal_frequencies", []))
    profile = np.asarray(rep.get("azimuthal_profile", []))
    spec_details = rep.get("spectral_details", {})
    comp = rep.get("spectral_comparison", {})

    fig, ax = plt.subplots(figsize=(6.4, 4.1))

    if len(freqs) == 0 or len(profile) == 0:
        ax.text(0.5, 0.5, "No radial spectral data", ha="center", va="center")
        fig.tight_layout()
        return fig

    # Highlight detected anomalous radial bands first
    anomaly_bands = rep.get("fft_anomaly_bands", [])
    for b_idx, (b_start, b_end) in enumerate(anomaly_bands):
        ax.axvspan(
            b_start,
            max(b_end, b_start + 1),
            color="#fca5a5",
            alpha=0.28,
            label="Anomalous High-Freq Band" if b_idx == 0 else None,
        )

    # Empirical real baseline (mean +/- std across reference real portraits)
    real_mean = comp.get("reference_real_mean")
    real_std = comp.get("reference_real_std")
    fake_mean = comp.get("reference_fake_mean")
    if real_mean is not None and len(real_mean) == len(freqs):
        ax.plot(freqs, real_mean, color="#059669", lw=1.8, linestyle="-", label=f"Authentic Baseline (n={comp.get('num_authentic_refs', 0)})")
        if real_std is not None and len(real_std) == len(freqs):
            ax.fill_between(freqs, real_mean - real_std, real_mean + real_std, color="#10b981", alpha=0.14)

    if fake_mean is not None and len(fake_mean) == len(freqs):
        ax.plot(freqs, fake_mean, color="#d97706", lw=1.6, linestyle="-.", label=f"Suspicious Ref Pattern (n={comp.get('num_suspicious_refs', 0)})")

    ideal_decay = comp.get("ideal_decay")
    if ideal_decay is None or len(ideal_decay) != len(freqs):
        base_val = float(profile[1]) if len(profile) > 1 else 1.0
        intercept = base_val + 1.69 * float(np.log1p(1.0))
        ideal_decay = np.maximum(0.0, intercept - 1.69 * np.log1p(np.maximum(freqs, 0.0)))
    ax.plot(freqs, ideal_decay, color="#94a3b8", linestyle="--", lw=1.3, label="Theoretical Optical Log Decay")

    # Query face profile
    ax.plot(freqs, profile, color="#1d4ed8", lw=2.3, label="Inspected Sample Profile")

    peak_idxs = spec_details.get("peak_indices", [])
    valid_peaks = [p for p in peak_idxs if 0 <= p < len(freqs)]
    if valid_peaks:
        ax.scatter(
            [freqs[p] for p in valid_peaks],
            [profile[p] for p in valid_peaks],
            color="#dc2626",
            s=62,
            zorder=5,
            label="Periodic Spikes",
        )

    all_vals = [profile]
    if real_mean is not None:
        all_vals.append(np.asarray(real_mean))
    if fake_mean is not None:
        all_vals.append(np.asarray(fake_mean))
    y_max = max(float(np.max(v)) for v in all_vals if len(v) > 0)
    y_min = min(float(np.min(v)) for v in all_vals if len(v) > 0)
    ax.set_ylim(y_min - 0.3, y_max + 0.16 * max(y_max - y_min, 1.0))

    ax.set_xlabel("Spatial Frequency Radius r (cycles / image)", fontsize=9.5)
    ax.set_ylabel("Mean Log Spectral Magnitude", fontsize=9.5)
    ax.set_title("Azimuthal Radial Power Spectrum (Sample vs Reference)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    return fig


def get_active_face_report(results: Dict[str, Any], key_prefix: str = "img") -> Dict[str, Any]:
    """Selects either the chosen multi-face sub-report or falls back cleanly to primary root results."""
    multi_faces = results.get("multi_face_reports", [])
    if len(multi_faces) > 1:
        options = [
            f"Face #{r['face_index'] + 1} ({'Primary' if r['is_primary'] else 'Secondary'}) — "
            f"{r['prediction_outcome']} ({r['fake_probability']:.3f})"
            for r in multi_faces
        ]
        sel_idx = st.selectbox(
            "Select Detected Face for Detailed XAI Inspection:",
            list(range(len(options))),
            format_func=lambda idx: options[idx],
            key=f"{key_prefix}_face_selector",
        )
        rep = dict(multi_faces[int(sel_idx)])
        # Ensure top-level root fields not duplicated on secondary faces still exist gracefully
        rep.setdefault("ela_comparison", results.get("ela_comparison", []))
        rep.setdefault("ela_limitations", results.get("ela_limitations", []))
        rep.setdefault("fft_anomaly_bands", results.get("fft_anomaly_bands", []))
        return rep
    return results


def render_unified_forensic_panel(results: Dict[str, Any], active_rep: Dict[str, Any]) -> None:
    """
    Renders the Unified 8-Item Forensic Evidence Panel required for every inspected image:
      1. Primary model prediction
      2. Calibrated confidence
      3. Grad-CAM visualization
      4. FFT visualization
      5. Radial power spectrum
      6. ELA visualization
      7. Face detection results
      8. Forensic observations (with explicit non-proof caveat)
    """
    cal_meta = results.get("calibration_metadata", {})
    has_face = bool(active_rep.get("has_face", results.get("faces_count", 0) > 0))
    img_type = active_rep.get("image_type", "Human Face Portrait" if has_face else "General Media / Object")
    face_label = f"Face #{active_rep.get('face_index', 0) + 1}" if len(results.get("multi_face_reports", [])) > 1 else ("Primary Face" if has_face else "Localized Object")

    with st.expander("📋 Unified Forensic Evidence Panel (All 8 Required Signals)", expanded=True):
        st.caption(
            f"Consolidates Primary Classifier Prediction, Calibrated Confidence, Localization ({img_type}), Spatial Grad-CAM, "
            f"2D FFT Spectrum, Azimuthal Radial Power Comparison, ELA Residuals, and Structured Forensic Observations ({face_label}). "
            "Visualizations are interpretive diagnostic aids — never standalone proof of manipulation."
        )

        # Items 1, 2 & 7: Summary Metrics Row
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("1. Primary Model Prediction", active_rep.get("prediction_outcome", "N/A"))
        u2.metric(
            "2. Calibrated Confidence",
            f"{active_rep.get('confidence_percent', 0.0):.2f}%",
            delta=f"P(Manip)={active_rep.get('fake_probability', 0.5):.4f} | T={cal_meta.get('temperature', 1.0):.2f}",
            delta_color="off",
        )
        if has_face:
            loc_label = "7. Face Detection Results"
            loc_val = f"{results.get('faces_count', 1)} Face(s) Detected"
        else:
            loc_label = "7. Media Localization"
            loc_val = img_type.split("/")[0].strip()

        bbox = active_rep.get("bbox", (0, 0, 256, 256))
        u3.metric(
            loc_label,
            loc_val,
            delta=f"Active Region: {bbox[2]}x{bbox[3]}px",
            delta_color="off",
        )
        ela_det = active_rep.get("ela_details", {})
        u4.metric(
            "ELA Compression Tier",
            ela_det.get("inconsistency_level", "N/A").split("/")[0].strip(),
            delta=f"Mean ELA={active_rep.get('ela_mean_error', 0.0):.2f} | Quad={ela_det.get('quad_discrepancy_ratio', 0.0):.3f}",
            delta_color="off",
        )

        # Items 3, 4, 5, 6 & Annotated Localization Image
        v1, v2, v3, v4, v5 = st.columns(5)
        with v1:
            ann_img = results.get("full_image_annotated") or results.get("aligned_face")
            cap_v1 = "7. Face Localization" if has_face else f"7. Object Localization ({img_type.split('/')[0].strip()})"
            st.image(ann_img, caption=cap_v1, width="stretch")
        with v2:
            st.image(active_rep["gradcam_overlay"], caption="3. Grad-CAM Overlay", width="stretch")
        with v3:
            fft_vis = active_rep.get("fft_log_colormap_img") or active_rep["fft_spectrum_img"]
            st.image(fft_vis, caption="4. 2D FFT Log-Magnitude", width="stretch")
        with v4:
            fig_rad = plot_radial_spectrum_comparison(active_rep)
            st.pyplot(fig_rad)
            plt.close(fig_rad)
            st.caption("5. Azimuthal Radial Power Spectrum")
        with v5:
            ela_vis = active_rep.get("ela_heatmap_color") or active_rep["ela_image"]
            st.image(ela_vis, caption="6. ELA Heatmap (Q=90)", width="stretch")

        # Item 8: Objective Forensic Observations & Caveats
        obs_list: List[str] = results.get("forensic_observations", [])
        if obs_list:
            st.markdown("##### 📝 8. Forensic Observations (Interpretive Summary)")
            for item in obs_list:
                st.markdown(f"- {item}")


def render_spatial_gradcam_tab(active_rep: Dict[str, Any], multi_faces: List[Dict[str, Any]]) -> None:
    """Renders detailed Grad-CAM spatial maps, regional breakdown, and multi-face comparison gallery."""
    st.markdown("#### Spatial Class Activation Mapping (Grad-CAM)")
    st.caption(
        "Visualizes spatial attention gradients over the convolutional feature maps for the inspected face crop. "
        "Note: Highlights regions driving neural activations; not a physical forgery mask."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(active_rep["aligned_face"], caption="Aligned Face Crop (256x256)", width="stretch")
    with c2:
        st.image(active_rep["gradcam_overlay"], caption="Grad-CAM Overlay on Face Crop", width="stretch")
    with c3:
        fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
        im = ax_cam.imshow(active_rep["gradcam_heatmap"], cmap="jet", vmin=0.0, vmax=1.0)
        plt.axis("off")
        fig_cam.colorbar(im, ax=ax_cam, fraction=0.046, pad=0.04)
        st.pyplot(fig_cam)
        plt.close(fig_cam)

    landmarks = active_rep.get("spatial_landmarks", {})
    if landmarks:
        is_face = bool(landmarks.get("is_facial_context", True))
        if is_face:
            st.markdown("##### 📍 Regional Facial Landmark Activation Distribution")
            l1, l2, l3 = st.columns(3)
            l1.metric("Ocular / Eyes Region", f"{landmarks.get('ocular_score', 0.0)*100:.1f}%")
            l2.metric("Mid-Facial (Nose/Mouth)", f"{landmarks.get('mid_facial_score', 0.0)*100:.1f}%")
            l3.metric("Jawline / Boundary Seam", f"{landmarks.get('boundary_score', 0.0)*100:.1f}%")
            context_note = "facial anatomy"
        else:
            ctx_type = landmarks.get("context_type", "Object")
            st.markdown(f"##### 📍 Object Spatial Zone Activation Distribution ({ctx_type})")
            l1, l2, l3 = st.columns(3)
            l1.metric("Upper Structural Perimeter", f"{landmarks.get('upper_object_score', landmarks.get('ocular_score', 0.0))*100:.1f}%")
            l2.metric("Central Object Core", f"{landmarks.get('central_core_score', landmarks.get('mid_facial_score', 0.0))*100:.1f}%")
            l3.metric("Lower Base / Support", f"{landmarks.get('lower_base_score', landmarks.get('boundary_score', 0.0))*100:.1f}%")
            context_note = "salient object geometry"

        dom = landmarks.get("dominant_region", "N/A")
        st.info(
            f"**Dominant Model Attention Zone:** Spatial gradients focus primarily on **{dom}**. "
            f"Use this to verify whether the network examines relevant {context_note} rather than background clutter."
        )

    if len(multi_faces) > 1:
        st.markdown("##### 👥 Multi-Face Per-Subject Grad-CAM Overlays")
        cols_g = st.columns(min(len(multi_faces), 4))
        for idx, rep in enumerate(multi_faces):
            with cols_g[idx % len(cols_g)]:
                st.image(
                    rep["gradcam_overlay"],
                    caption=f"Face #{rep['face_index'] + 1}: {rep['prediction_outcome']} ({rep['fake_probability']:.3f})",
                    width="stretch",
                )


def render_spectral_fft_tab(active_rep: Dict[str, Any]) -> None:
    """Renders 2D FFT Magnitude, Log-Magnitude Colormap, Anomaly Regions & Real-vs-Suspicious Radial Spectrum."""
    st.markdown("#### Frequency Domain & Azimuthal Radial Decay Analysis")
    st.caption(
        "Displays shifted 2D FFT magnitude spectra, pseudo-color log-magnitude distribution, "
        "highlighted high-frequency anomaly regions, and real-vs-suspicious radial power decay comparisons."
    )

    spec_details = active_rep.get("spectral_details", {})
    comp = active_rep.get("spectral_comparison", {})

    f1, f2, f3 = st.columns(3)
    with f1:
        st.image(active_rep["fft_spectrum_img"], caption="2D FFT Magnitude Spectrum (Grayscale)", width="stretch")
    with f2:
        log_vis = active_rep.get("fft_log_colormap_img") or active_rep["fft_spectrum_img"]
        st.image(log_vis, caption="Log-Magnitude Spectrum (Inferno Scale)", width="stretch")
    with f3:
        anom_vis = active_rep.get("fft_anomaly_map") or active_rep["fft_spectrum_img"]
        st.image(anom_vis, caption="Frequency Anomaly Regions (Annular Highlight)", width="stretch")

    st.markdown("##### 📈 Azimuthal Radial Power Spectrum (Real vs Suspicious Reference Comparison)")
    g_col, m_col = st.columns([1.45, 1.0])
    with g_col:
        fig_az = plot_radial_spectrum_comparison(active_rep)
        st.pyplot(fig_az)
        plt.close(fig_az)
    with m_col:
        has_spikes = spec_details.get("has_deconv_spikes", False)
        st.metric("Periodic Grid Anomaly", "Spikes Detected" if has_spikes else "Nominal")
        st.metric("High-Frequency Energy Ratio", f"{spec_details.get('high_freq_ratio', 0.0)*100:.1f}%")
        st.metric("Spectral Smoothness (Var Δ²)", f"{spec_details.get('spectral_smoothness', 0.0):.4f}")
        st.metric("Excess vs Reference Baseline", f"{comp.get('high_freq_excess_pct', 0.0):+.2f}%")
        if comp.get("comparison_summary"):
            st.info(comp["comparison_summary"])

    st.caption(
        "⚖️ **Spectral Interpretation Limitation:** Elevated high-frequency FFT energy or radial ripples can stem from "
        "generative upsampling checkerboards OR benign post-processing (camera sharpening, resizing, WebP/JPEG artifacts). "
        "Frequency patterns are corroborative signals rather than definitive proof."
    )


def render_ela_tab(active_rep: Dict[str, Any], root_results: Dict[str, Any]) -> None:
    """Renders ELA heatmaps, JPEG recompression comparisons (Q=95/90/75), inconsistency stats & explicit caveats."""
    st.markdown("#### Error Level Analysis (JPEG Quantization Residuals)")
    st.caption(
        "Evaluates local compression residual differences under controlled JPEG recompression. "
        "Reported as an explanatory signal only — see explicit ELA limitations below."
    )

    ela_det = active_rep.get("ela_details", {})
    e1, e2, e3 = st.columns(3)
    with e1:
        st.image(active_rep["aligned_face"], caption="Aligned Subject Crop", width="stretch")
    with e2:
        st.image(active_rep["ela_image"], caption=f"Enhanced ELA (Q=90, Scale 15x, Mean: {active_rep['ela_mean_error']:.2f})", width="stretch")
    with e3:
        ela_color = active_rep.get("ela_heatmap_color")
        if ela_color is not None:
            st.image(ela_color, caption="ELA Thermal Heatmap (Inferno Scale)", width="stretch")
        else:
            diff_gray = ela_det.get("diff_np_gray")
            if diff_gray is not None:
                fig_ela, ax_ela = plt.subplots(figsize=(4, 4))
                im_ela = ax_ela.imshow(diff_gray, cmap="inferno")
                plt.axis("off")
                fig_ela.colorbar(im_ela, ax=ax_ela, fraction=0.046, pad=0.04)
                st.pyplot(fig_ela)
                plt.close(fig_ela)

    st.markdown("##### 📊 Compression Inconsistency Report & Histogram")
    st.info(
        f"**Compression Inconsistency Assessment:** `{ela_det.get('inconsistency_level', 'N/A')}` — "
        f"{ela_det.get('inconsistency_summary', '')}"
    )

    h_col1, h_col2 = st.columns([1, 1])
    with h_col1:
        hist_counts = ela_det.get("hist_counts", [])
        bin_edges = ela_det.get("bin_edges", [])
        if hist_counts and bin_edges:
            fig_h, ax_h = plt.subplots(figsize=(5, 3.2))
            ax_h.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges), edgecolor="black", color="#3b82f6", alpha=0.75)
            ax_h.set_title("Pixel Difference Error Histogram", fontsize=10, fontweight="bold")
            ax_h.set_xlabel("Error Magnitude", fontsize=9)
            ax_h.set_ylabel("Pixel Frequency", fontsize=9)
            ax_h.grid(True, linestyle=":", alpha=0.6)
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
    with h_col2:
        q_errs = ela_det.get("quadrant_errors", {})
        st.markdown("**Regional Quadrant Mean Error:**")
        st.write(f"- Top-Left (Hair / Forehead): `{q_errs.get('top_left', 0.0):.2f}`")
        st.write(f"- Top-Right (Hair / Background): `{q_errs.get('top_right', 0.0):.2f}`")
        st.write(f"- Bottom-Left (Jawline / Chin): `{q_errs.get('bottom_left', 0.0):.2f}`")
        st.write(f"- Bottom-Right (Neck / Collar): `{q_errs.get('bottom_right', 0.0):.2f}`")
        st.write(
            f"**Quadrant Discrepancy Ratio:** `{ela_det.get('quad_discrepancy_ratio', 0.0):.3f}` &nbsp;|&nbsp; "
            f"**16x16 Block Inconsistency CV:** `{ela_det.get('block_inconsistency_cv', 0.0):.3f}`"
        )

    ela_comps = active_rep.get("ela_comparison") or root_results.get("ela_comparison", [])
    if ela_comps:
        st.markdown("##### 🔁 Multi-Quality JPEG Recompression Comparisons (Q=95 vs Q=90 vs Q=75)")
        q_cols = st.columns(len(ela_comps))
        for idx, item in enumerate(ela_comps):
            with q_cols[idx]:
                vis = item.get("ela_heatmap_color") or item["ela_image"]
                st.image(
                    vis,
                    caption=(
                        f"Recompressed Q={item['quality']} | "
                        f"Mean={item['mean_error']:.2f}, Quad={item['quad_discrepancy_ratio']:.3f}"
                    ),
                    width="stretch",
                )

    limitations = active_rep.get("ela_limitations") or root_results.get("ela_limitations") or ela_det.get("limitations", [])
    if limitations:
        with st.expander("⚠️ Known Methodological Limitations of Error Level Analysis (ELA)", expanded=True):
            for lim in limitations:
                st.markdown(f"- {lim}")


def render_srm_tab(active_rep: Dict[str, Any]) -> None:
    """Renders Spatial Rich Model (SRM) high-pass noise residuals and texture inconsistency metrics."""
    st.markdown("#### Spatial Noise Residual & Steganalysis Analysis (SRM)")
    st.caption(
        "Extracts spatial noise residuals using 3 standardized SRM high-pass kernels (F1 Directional, F2 Laplacian, F3 Square Edge). "
        "Exposes localized noise inconsistencies, unnatural texture smoothing, and synthetic generator footprints."
    )
    srm_det = active_rep.get("srm_details")
    if not srm_det:
        st.info("Spatial SRM residual data is not available for this record.")
        return

    s1, s2 = st.columns([1, 1.4])
    with s1:
        if srm_det.get("heatmap_image"):
            st.image(srm_det["heatmap_image"], caption="SRM High-Pass Residual Thermal Map (Viridis Scale)", width="stretch")
    with s2:
        st.markdown(f"**Consistency Assessment:** `{srm_det.get('consistency_tier', 'Analyzed')}`")
        st.write(srm_det.get("consistency_summary", ""))
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Mean Residual Energy", f"{srm_det.get('mean_energy', 0.0):.3f}")
        m2.metric("16x16 Block Noise CV", f"{srm_det.get('block_inconsistency_cv', 0.0):.3f}")
        m3, m4 = st.columns(2)
        m3.metric("Residual Kurtosis", f"{srm_det.get('residual_kurtosis', 0.0):.2f}", delta="Gaussian Baseline ~3.0", delta_color="off")
        m4.metric("Quadrant Discrepancy", f"{srm_det.get('quad_discrepancy_ratio', 0.0):.3f}")

    st.caption(
        "🔬 **Forensic Steganalysis Guidance:** Real camera sensors produce uniform Poisson-Gaussian noise across identical textures. "
        "High block CV or anomalous residual kurtosis points to localized composite boundaries or neural synthesis smoothing."
    )
