"""
Unified End-to-End Forensic Inference Pipeline.
Integrates face alignment, dual-domain forward pass, Grad-CAM heatmap generation,
2D FFT azimuthal radial profiling, and Error Level Analysis (ELA).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.core.face_extractor import FaceExtractor
from src.core.utils import (
    SPATIAL_TRANSFORM,
    SPECTRAL_TRANSFORM,
    compute_throughput_fps,
    get_device,
    get_runtime_memory_telemetry,
    load_config,
    safe_batched_forward,
)
from src.evaluation.calibration import (
    OUTCOME_AUTHENTIC,
    OUTCOME_FAKE,
    OUTCOME_MANIPULATED,
    OUTCOME_REAL,
    OUTCOME_UNCERTAIN,
    classify_with_policy,
    default_uncertainty_policy,
    load_calibration_policy,
    sigmoid_np,
)
from src.explainability.ela import ELA_FORENSIC_LIMITATIONS, ErrorLevelAnalysis
from src.explainability.gradcam import GradCAM, compute_regional_scores
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.explainability.srm_analysis import SpatialSRMAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet


class ForensicInferencePipeline:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: str = "config/default_config.yaml",
        device: str = "auto",
        calibration_path: Optional[str] = "results/calibration/calibration_results.json",
        use_default_threshold: bool = False,
    ):
        self.config = load_config(config_path) if Path(config_path).exists() else {}
        self.device = get_device(device)
        
        # Modules
        self.face_extractor = FaceExtractor(target_size=(256, 256))
        self.spectral_analyzer = SpectralAnalyzer(target_size=256)
        self.ela_analyzer = ErrorLevelAnalysis(
            quality=self.config.get("forensics", {}).get("ela_quality", 90),
            scale=self.config.get("forensics", {}).get("ela_scale", 15)
        )
        self.srm_analyzer = SpatialSRMAnalyzer()
        
        # Initialize Neural Network
        self.model = DualDomainSentinelNet(
            backbone_name=self.config.get("model", {}).get("backbone", "efficientnet_b0"),
            pretrained=False,  # Weights loaded from checkpoint or initialized
            spatial_dim=self.config.get("model", {}).get("spatial_feat_dim", 512),
            spectral_dim=self.config.get("model", {}).get("spectral_feat_dim", 256),
            fusion_type=self.config.get("model", {}).get("fusion_type", "cross_attention")
        ).to(self.device)
        
        # Load weights if checkpoint exists
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[✓] Loaded forensic model checkpoint: {checkpoint_path}")
        else:
            self.model.eval()
            print("[i] Running with initialized backbone weights.")

        # Validation-derived calibration parameters (Temperature + 3-outcome policy)
        if calibration_path:
            self.calibration_info = load_calibration_policy(calibration_path)
        else:
            self.calibration_info = {
                "calibrated": False,
                "calibration_path": None,
                "temperature": 1.0,
                **default_uncertainty_policy(),
            }
        self.temperature = float(self.calibration_info.get("temperature", 1.0))
        self.uncertainty_policy = {
            "decision_threshold": float(self.calibration_info.get("decision_threshold", 0.50)),
            "uncertain_low": float(self.calibration_info.get("uncertain_low", 0.40)),
            "uncertain_high": float(self.calibration_info.get("uncertain_high", 0.60)),
            "balanced_threshold": float(self.calibration_info.get("balanced_threshold", self.calibration_info.get("decision_threshold", 0.71))),
            "high_recall_threshold": float(self.calibration_info.get("high_recall_threshold", 0.34)),
            "high_precision_threshold": float(self.calibration_info.get("high_precision_threshold", 0.72)),
            "min_laplacian_sharpness": float(self.calibration_info.get("min_laplacian_sharpness", 15.0)),
            "min_dimension_px": int(self.calibration_info.get("min_dimension_px", 80)),
        }
        if use_default_threshold:
            self.uncertainty_policy["decision_threshold"] = 0.50
            
        # Initialize Grad-CAM
        target_layer = self.model.spatial_stream.target_cam_layer
        self.grad_cam = GradCAM(model=self.model, target_layer=target_layer)

    def _evaluate_auxiliary_forensics(
        self,
        az_profile: np.ndarray,
        spectral_details: Dict[str, Any],
        ela_details: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Computes decoupled auxiliary frequency and ELA anomaly indicators for explanatory telemetry.
        These scores are strictly separated from the primary classifier and never override neural logits.
        """
        # A. Spectral 2D FFT Anomaly Indicator
        peak_count = len(spectral_details.get("detected_peaks", []))
        if len(az_profile) > 20:
            norm_prof = az_profile / (np.max(az_profile) + 1e-6)
            diff2 = np.abs(np.diff(norm_prof, n=2))
            peak_spike = float(np.max(diff2[len(diff2) // 3:]))
            if peak_count >= 6 and peak_spike > 0.075:
                p_spectral = min(0.95, 0.50 + (peak_spike - 0.075) * 5.0)
            else:
                p_spectral = max(0.06, 0.14 * (peak_spike / 0.075))
        else:
            p_spectral = 0.15

        # B. Compression Quantization Anomaly Indicator (ELA)
        mean_ela_error = float(ela_details.get("mean_error", 0.0))
        quad_mismatch = float(ela_details.get("quad_discrepancy_ratio", 0.0))
        if quad_mismatch > 0.55 and mean_ela_error > 3.0:
            p_ela = min(0.95, 0.40 + (quad_mismatch - 0.55) * 1.8)
        else:
            p_ela = max(0.06, 0.10 + 0.12 * min(quad_mismatch / 0.55, 1.5))

        return float(p_spectral), float(p_ela)

    @staticmethod
    def _format_three_outcome_verdict(outcome_res: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        """Maps a 3-outcome policy dict to user-facing labels, badges, CSS classes, and explanations."""
        outcome = outcome_res.get("prediction_outcome", OUTCOME_UNCERTAIN)
        tau = float(outcome_res.get("decision_threshold", 0.71))
        u_low = float(outcome_res.get("uncertain_low", 0.69))
        u_high = float(outcome_res.get("uncertain_high", 0.73))

        if outcome in (OUTCOME_FAKE, "MANIPULATED"):
            return (
                OUTCOME_FAKE,
                "Manipulated / AI-Generated",
                "🚨 VERDICT: FAKE",
                "verdict-fake",
                f"Calibrated classifier probability exceeded the decision boundary ({u_high:.3f}). Review Grad-CAM, FFT, SRM, and ELA evidence alongside file provenance.",
            )
        if outcome in (OUTCOME_REAL, "AUTHENTIC"):
            return (
                OUTCOME_REAL,
                "Authentic Media",
                "✅ VERDICT: REAL",
                "verdict-real",
                f"Calibrated classifier probability fell below the authenticity boundary ({u_low:.3f}). Patterns align with optical capture statistics.",
            )
        reason = outcome_res.get("uncertainty_reason") or (
            f"Prediction falls within validated ambiguity zone [{u_low:.3f}, {u_high:.3f}] around threshold {tau:.3f}."
        )
        return (
            OUTCOME_UNCERTAIN,
            "Uncertain / Manual Review",
            "⚖️ VERDICT: UNCERTAIN",
            "verdict-inconclusive",
            f"{reason} Escalate to manual forensic review rather than relying on an automated binary call.",
        )

    def _prepare_face_features(
        self,
        f_rec: Dict[str, Any],
        include_ela_comparison: bool = True,
        container_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extracts FFT spectral features, ELA residuals, SRM spatial noise, sharpness, and model-ready CPU tensors for one face."""
        aligned_face = f_rec["aligned_face"]
        face_np = np.array(aligned_face.convert("RGB"))
        object_bbox = f_rec.get("object_bbox")

        # A. 2D FFT Spectral Analysis & Real-vs-Suspicious Comparison
        t_start_fft = time.time()
        mag_spec, raw_log_spec = self.spectral_analyzer.compute_fft_magnitude(aligned_face)
        az_freqs, az_profile = self.spectral_analyzer.azimuthal_average(raw_log_spec)
        spectral_details = self.spectral_analyzer.analyze_spectral_peaks(az_freqs, az_profile)
        fft_log_colormap_img = self.spectral_analyzer.compute_colormap_log_spectrum(raw_log_spec)
        fft_anomaly_map, fft_anomaly_bands = self.spectral_analyzer.compute_frequency_anomaly_map(
            raw_log_spec, az_freqs, spectral_details
        )
        spectral_comparison = self.spectral_analyzer.compare_real_vs_suspicious(
            az_freqs, az_profile, spectral_details
        )
        t_fft = (time.time() - t_start_fft) * 1000.0

        # B. Error Level Analysis (ELA) & Recompression Comparison
        t_start_ela = time.time()
        ela_details = self.ela_analyzer.compute_ela_detailed(
            aligned_face, container_format=container_format, object_bbox=object_bbox
        )
        ela_comparison = (
            self.ela_analyzer.compute_recompression_comparison(aligned_face, qualities=(95, 90, 75))
            if include_ela_comparison
            else []
        )
        t_ela = (time.time() - t_start_ela) * 1000.0

        # C. Spatial SRM Noise Residual Analysis
        t_start_srm = time.time()
        srm_details = self.srm_analyzer.analyze_residuals(aligned_face)
        t_srm = (time.time() - t_start_srm) * 1000.0

        # D. Model-ready CPU tensors (stacked later for batched forward pass)
        t_start_tensor = time.time()
        spec_img = Image.fromarray(mag_spec)
        spatial_tensor = SPATIAL_TRANSFORM(aligned_face)
        spectral_tensor = SPECTRAL_TRANSFORM(spec_img)
        t_prep_tensor = (time.time() - t_start_tensor) * 1000.0

        # E. Auxiliary Scores & Edge Sharpness
        p_spectral, p_ela = self._evaluate_auxiliary_forensics(az_profile, spectral_details, ela_details)
        gray_crop = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

        return {
            "f_rec": f_rec,
            "aligned_face": aligned_face,
            "face_np": face_np,
            "spec_img": spec_img,
            "spatial_tensor": spatial_tensor,
            "spectral_tensor": spectral_tensor,
            "fft_log_colormap_img": fft_log_colormap_img,
            "fft_anomaly_map": fft_anomaly_map,
            "fft_anomaly_bands": fft_anomaly_bands,
            "az_freqs": az_freqs,
            "az_profile": az_profile,
            "spectral_details": spectral_details,
            "spectral_comparison": spectral_comparison,
            "ela_details": ela_details,
            "ela_comparison": ela_comparison,
            "srm_details": srm_details,
            "p_spectral": p_spectral,
            "p_ela": p_ela,
            "laplacian_variance": laplacian_var,
            "t_fft": t_fft,
            "t_ela": t_ela,
            "t_srm": t_srm,
            "t_prep_tensor": t_prep_tensor,
        }

    def _finalize_face_report(
        self,
        prep: Dict[str, Any],
        raw_logit: float,
        active_policy: Dict[str, Any],
        orig_image_size: Tuple[int, int],
        t_neural_share: float,
        operating_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """Applies calibration policy and computes per-face Grad-CAM attribution after batched forward inference."""
        f_rec = prep["f_rec"]
        aligned_face = prep["aligned_face"]
        face_np = prep["face_np"]
        laplacian_var = prep["laplacian_variance"]
        min_lap = float(active_policy.get("min_laplacian_sharpness", 15.0))
        min_dim = int(active_policy.get("min_dimension_px", 80))

        p_neural_uncalibrated = float(sigmoid_np(np.array([raw_logit]))[0])
        fake_prob = float(sigmoid_np(np.array([raw_logit / max(self.temperature, 1e-6)]))[0])
        real_prob = 1.0 - fake_prob

        # Decoupled branch prediction & cross-domain conflict analysis
        sp_1 = prep["spatial_tensor"].unsqueeze(0).to(self.device)
        fr_1 = prep["spectral_tensor"].unsqueeze(0).to(self.device)
        branch_info = None
        try:
            branch_info = self.model.compute_branch_predictions(
                sp_1, fr_1, temperature=self.temperature
            )
        except Exception:
            branch_info = None

        quality_reasons = []
        if laplacian_var < min_lap:
            quality_reasons.append(f"Edge sharpness variance ({laplacian_var:.1f}) below forensic minimum ({min_lap:.1f})")
        if min(orig_image_size) < min_dim:
            quality_reasons.append(f"Image resolution {orig_image_size[0]}x{orig_image_size[1]} below minimum ({min_dim}px)")
        quality_ok = len(quality_reasons) == 0
        quality_reason = "; ".join(quality_reasons) if quality_reasons else None

        outcome_res = classify_with_policy(
            calibrated_prob=fake_prob,
            policy=active_policy,
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            branch_info=branch_info,
            operating_mode=operating_mode,
        )
        label, verdict_tier, verdict_badge, verdict_class, verdict_explanation = self._format_three_outcome_verdict(outcome_res)

        # Per-face Grad-CAM backward pass (target_class reflects predicted class)
        t_start_cam = time.time()
        target_cls = 1 if fake_prob >= 0.50 else 0
        cam_map = self.grad_cam.generate_cam(sp_1, fr_1, target_class=target_cls)
        gradcam_overlay_np = self.grad_cam.overlay_cam(face_np, cam_map, alpha=0.45)
        has_face_flag = bool(f_rec.get("has_face", f_rec.get("face_detected", True)))
        img_type_ctx = f_rec.get("image_type")
        spatial_landmarks = compute_regional_scores(cam_map, has_face=has_face_flag, image_type=img_type_ctx)
        t_gradcam = (time.time() - t_start_cam) * 1000.0

        ela_details = prep["ela_details"]
        srm_details = prep["srm_details"]
        return {
            "face_index": int(f_rec["face_index"]),
            "is_primary": bool(f_rec["is_primary"]),
            "bbox": f_rec["bbox"],
            "raw_face_bbox": f_rec.get("raw_face_bbox"),
            "face_detected": bool(f_rec["face_detected"]),
            "aligned_face": aligned_face,
            "aligned_face_np": face_np,
            "raw_logit": float(raw_logit),
            "uncalibrated_prob": p_neural_uncalibrated,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence_percent": float(outcome_res["classifier_confidence"]) * 100.0,
            "prediction_label": label,
            "prediction_outcome": outcome_res["prediction_outcome"],
            "is_fake": bool(outcome_res["is_fake"]),
            "is_uncertain": bool(outcome_res["is_uncertain"]),
            "uncertainty_reason": outcome_res.get("uncertainty_reason"),
            "verdict_tier": verdict_tier,
            "verdict_badge": verdict_badge,
            "verdict_class": verdict_class,
            "verdict_explanation": verdict_explanation,
            "outcome_res": outcome_res,
            "spatial_fake_prob": float(branch_info["spatial_prob"]) if branch_info else None,
            "spatial_logit": float(branch_info["spatial_logit"]) if branch_info else None,
            "frequency_fake_prob": float(branch_info["frequency_prob"]) if branch_info else None,
            "frequency_logit": float(branch_info["frequency_logit"]) if branch_info else None,
            "branch_discrepancy": float(branch_info["discrepancy"]) if branch_info else 0.0,
            "has_branch_conflict": bool(branch_info["has_conflict"]) if branch_info else False,
            "false_negative_risk": bool(branch_info["false_negative_risk"]) if branch_info else False,
            "false_negative_warning": outcome_res.get("false_negative_warning"),
            "evidence_consistency": outcome_res.get("evidence_consistency", "CONSISTENT"),
            "operating_mode": operating_mode,
            "branch_info": branch_info,
            "confidence_percent": float(outcome_res["classifier_confidence"]) * 100.0,
            "prediction_label": label,
            "prediction_outcome": outcome_res["prediction_outcome"],
            "is_fake": bool(outcome_res["is_fake"]),
            "is_uncertain": bool(outcome_res["is_uncertain"]),
            "uncertainty_reason": outcome_res.get("uncertainty_reason"),
            "verdict_tier": verdict_tier,
            "verdict_badge": verdict_badge,
            "verdict_class": verdict_class,
            "verdict_explanation": verdict_explanation,
            "outcome_res": outcome_res,
            "gradcam_heatmap": cam_map,
            "gradcam_overlay": Image.fromarray(gradcam_overlay_np),
            "spatial_landmarks": spatial_landmarks,
            "fft_spectrum_img": prep["spec_img"],
            "fft_log_colormap_img": prep["fft_log_colormap_img"],
            "fft_anomaly_map": prep["fft_anomaly_map"],
            "fft_anomaly_bands": prep["fft_anomaly_bands"],
            "azimuthal_frequencies": prep["az_freqs"],
            "azimuthal_profile": prep["az_profile"],
            "spectral_details": prep["spectral_details"],
            "spectral_comparison": prep["spectral_comparison"],
            "ela_image": ela_details["ela_image"],
            "ela_heatmap_color": ela_details["ela_heatmap_color"],
            "ela_mean_error": ela_details["mean_error"],
            "ela_details": ela_details,
            "ela_comparison": prep["ela_comparison"],
            "ela_limitations": ela_details.get("limitations", list(ELA_FORENSIC_LIMITATIONS)),
            "p_spectral": prep["p_spectral"],
            "p_ela": prep["p_ela"],
            "laplacian_variance": laplacian_var,
            "t_fft": prep["t_fft"],
            "t_ela": prep["t_ela"],
            "t_neural": t_neural_share,
            "t_gradcam": t_gradcam,
        }

    def _analyze_face_record(
        self,
        f_rec: Dict[str, Any],
        active_policy: Dict[str, Any],
        orig_image_size: Tuple[int, int],
        include_ela_comparison: bool = True,
        operating_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """Single-face compatibility helper executing prepare -> forward -> finalize."""
        prep = self._prepare_face_features(f_rec=f_rec, include_ela_comparison=include_ela_comparison)
        t_start_neural = time.time()
        with torch.no_grad():
            logits, active_dev, _ = safe_batched_forward(
                self.model,
                prep["spatial_tensor"].unsqueeze(0),
                prep["spectral_tensor"].unsqueeze(0),
                self.device,
            )
            self.device = active_dev
            raw_logit = float(logits.reshape(-1)[0].item())
        t_neural = (time.time() - t_start_neural) * 1000.0
        return self._finalize_face_report(
            prep=prep,
            raw_logit=raw_logit,
            active_policy=active_policy,
            orig_image_size=orig_image_size,
            t_neural_share=t_neural,
            operating_mode=operating_mode,
        )

    @staticmethod
    def _annotate_full_image(image: Image.Image, face_reports: List[Dict[str, Any]]) -> Image.Image:
        """Draws color-coded face detection boxes and per-face calibrated verdicts on a copy of the source image."""
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        color_map = {
            OUTCOME_AUTHENTIC: "#10b981",
            OUTCOME_MANIPULATED: "#ef4444",
            OUTCOME_UNCERTAIN: "#f59e0b",
        }
        for rep in face_reports:
            x, y, w, h = rep["bbox"]
            outcome = rep["prediction_outcome"]
            box_color = color_map.get(outcome, "#94a3b8") if rep["face_detected"] else "#94a3b8"
            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=3)
            tag = (
                f"Face #{rep['face_index'] + 1}: {outcome.split(' /')[0]} ({rep['fake_probability']:.2f})"
                if rep["face_detected"]
                else f"Center Crop Fallback ({rep['fake_probability']:.2f})"
            )
            draw.rectangle([x, max(0, y - 18), min(canvas.width, x + len(tag) * 6 + 10), max(18, y)], fill=box_color)
            draw.text((x + 4, max(2, y - 15)), tag, fill="#ffffff")
        return canvas

    @staticmethod
    def _build_forensic_observations(
        primary: Dict[str, Any],
        faces_detected_count: int,
        cal_meta: Dict[str, Any],
    ) -> List[str]:
        """Generates objective, evidence-grounded observations for the Unified Forensic Panel."""
        obs: List[str] = []
        obs.append(
            f"Primary Classifier: {primary['prediction_outcome']} with calibrated confidence "
            f"{primary['confidence_percent']:.1f}% (P(Manipulated)={primary['fake_probability']:.4f}, "
            f"T={cal_meta['temperature']:.2f}, tau={cal_meta['decision_threshold']:.2f})."
        )
        # Cross-Domain Telemetry Observation
        fn_risk = bool(primary.get("false_negative_risk", False))
        fn_warn = primary.get("false_negative_warning")
        has_conflict = bool(primary.get("has_branch_conflict", False))
        disc = float(primary.get("branch_discrepancy", 0.0))
        s_prob = primary.get("spatial_fake_prob")
        f_prob = primary.get("frequency_fake_prob")
        if fn_risk and fn_warn:
            obs.append(f"⚠️ {fn_warn}")
        elif s_prob is not None and f_prob is not None:
            if has_conflict:
                obs.append(
                    f"Cross-Domain Discrepancy: Spatial branch P(Fake)={s_prob*100:.1f}%, Frequency branch P(Fake)={f_prob*100:.1f}% "
                    f"(divergence={disc*100:.1f}%). Multi-domain agreement is divided."
                )
            else:
                obs.append(
                    f"Cross-Domain Corroboration: Spatial branch ({s_prob*100:.1f}% Fake) and Frequency branch "
                    f"({f_prob*100:.1f}% Fake) are mutually consistent (discrepancy={disc*100:.1f}%)."
                )
        has_face = bool(primary.get("has_face", faces_detected_count > 0))
        img_type = primary.get("image_type", "Human Face Portrait" if has_face else "General Media / Object")
        if has_face and faces_detected_count > 0:
            obs.append(
                f"Face Localization: Detected {faces_detected_count} verifiable human face(s); primary facial region "
                f"({primary['bbox'][2]}x{primary['bbox'][3]}px, Laplacian sharpness={primary['laplacian_variance']:.1f})."
            )
        else:
            obs.append(
                f"Media Localization: Identified as '{img_type}'. No frontal human face confirmed; evaluated "
                f"object-aware localized region ({primary['bbox'][2]}x{primary['bbox'][3]}px, Laplacian sharpness={primary['laplacian_variance']:.1f})."
            )

        dom_region = primary["spatial_landmarks"].get("dominant_region", "N/A")
        obs.append(
            f"Spatial Attention (Grad-CAM): Peak activation concentrated in {dom_region}. "
            "Grad-CAM reflects neural network focus rather than localized physical manipulation proof."
        )

        spec_det = primary["spectral_details"]
        bands = primary.get("fft_anomaly_bands", [])
        band_txt = ", ".join(f"r={b[0]}" if b[0] == b[1] else f"r=[{b[0]}..{b[1]}]" for b in bands[:3]) if bands else "none"
        obs.append(
            f"Frequency Spectrum (2D FFT): {'Detected ' + str(len(spec_det.get('detected_peaks', []))) + ' periodic spike(s)' if spec_det.get('has_deconv_spikes') else 'Smooth radial decay'}; "
            f"anomalous frequency bands: {band_txt}; excess energy {primary['spectral_comparison'].get('high_freq_excess_pct', 0.0):+.1f}% vs reference baseline."
        )

        srm_det = primary.get("srm_details", {})
        if srm_det:
            obs.append(
                f"Spatial Noise Residuals (SRM): {srm_det.get('consistency_tier', 'Analyzed')} "
                f"(mean energy={srm_det.get('mean_energy', 0.0):.2f}, block CV={srm_det.get('block_inconsistency_cv', 0.0):.3f}, kurtosis={srm_det.get('residual_kurtosis', 0.0):.1f})."
            )

        ela_det = primary["ela_details"]
        ela_fmt = ela_det.get("container_format", "JPEG")
        obs.append(
            f"Compression Residuals (ELA Q=90, format={ela_fmt}): {ela_det.get('inconsistency_level', 'N/A')} "
            f"(mean error={primary['ela_mean_error']:.2f}, quad ratio={ela_det.get('quad_discrepancy_ratio', 0.0):.3f}). "
            "Subject to format quantization characteristics and natural edge response."
        )
        obs.append(
            "Interpretive Safeguard: Visual artifacts (Grad-CAM heatmaps, FFT spectra, SRM residuals, and ELA maps) are "
            "exploratory diagnostic aids and NEVER constitute definitive standalone proof of manipulation."
        )
        return obs

    def analyze_image(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
        operating_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Runs comprehensive forensic examination on an image.

        Uses temperature-scaled neural probability directly (zero Bayesian/heuristic override),
        applies the validation-derived 3-outcome uncertainty policy, generates per-face
        Grad-CAM, FFT & ELA visualizations, and keeps auxiliary telemetry decoupled.
        Supports operating modes: 'balanced' (default), 'high_recall', and 'high_precision'.
        """
        start_time = time.time()

        # 1. Face Extraction & Alignment (Supports Multi-Face & Fallback)
        t_start_face = time.time()
        all_face_records = self.face_extractor.extract_all_faces(image)
        t_face = (time.time() - t_start_face) * 1000.0

        active_policy = dict(self.uncertainty_policy)
        if threshold is not None:
            t_req = float(np.clip(threshold, 0.05, 0.95))
            lower_m = max(0.01, active_policy["decision_threshold"] - active_policy["uncertain_low"])
            upper_m = max(0.01, active_policy["uncertain_high"] - active_policy["decision_threshold"])
            active_policy["decision_threshold"] = t_req
            active_policy["uncertain_low"] = max(0.01, round(t_req - lower_m, 4))
            active_policy["uncertain_high"] = min(0.99, round(t_req + upper_m, 4))
        elif operating_mode == "high_recall":
            tau = float(active_policy.get("high_recall_threshold", 0.34))
            active_policy["decision_threshold"] = tau
            active_policy["uncertain_low"] = max(0.01, round(tau - 0.05, 4))
            active_policy["uncertain_high"] = min(0.99, round(tau + 0.05, 4))
        elif operating_mode == "high_precision":
            tau = float(active_policy.get("high_precision_threshold", 0.72))
            active_policy["decision_threshold"] = tau
            active_policy["uncertain_low"] = max(0.01, round(tau - 0.015, 4))
            active_policy["uncertain_high"] = min(0.99, round(tau + 0.015, 4))

        # 2. Prepare FFT/ELA/SRM features & tensors across all detected faces (vectorized batch prep)
        container_fmt = getattr(image, "format", None)
        prepared_faces: List[Dict[str, Any]] = [
            self._prepare_face_features(
                f_rec=f_rec,
                include_ela_comparison=(idx == 0),
                container_format=container_fmt,
            )
            for idx, f_rec in enumerate(all_face_records)
        ]

        # 3. Single Batched Dual-Domain Neural Forward Pass across all extracted faces
        self.model.eval()
        spatial_batch = torch.stack([p["spatial_tensor"] for p in prepared_faces])
        spectral_batch = torch.stack([p["spectral_tensor"] for p in prepared_faces])
        t_start_neural = time.time()
        with torch.no_grad():
            logits_tensor, active_dev, fallback_triggered = safe_batched_forward(
                self.model,
                spatial_batch,
                spectral_batch,
                self.device,
                enable_cpu_fallback=True,
            )
            self.device = active_dev
            raw_logits = logits_tensor.reshape(-1).detach().cpu().numpy().tolist()
        t_neural_batch_ms = (time.time() - t_start_neural) * 1000.0
        per_face_neural_ms = t_neural_batch_ms / max(len(prepared_faces), 1)

        # 4. Finalize calibrated 3-outcome verdicts & per-face Grad-CAM attribution
        analyzed_faces: List[Dict[str, Any]] = [
            self._finalize_face_report(
                prep=prep,
                raw_logit=float(logit),
                active_policy=active_policy,
                orig_image_size=image.size,
                t_neural_share=per_face_neural_ms,
                operating_mode=operating_mode,
            )
            for prep, logit in zip(prepared_faces, raw_logits)
        ]

        primary = analyzed_faces[0]
        faces_detected_count = len([f for f in all_face_records if f["face_detected"]])
        full_image_annotated = self._annotate_full_image(image, analyzed_faces)

        cal_meta = {
            "calibrated": bool(self.calibration_info.get("calibrated", False)),
            "calibration_path": self.calibration_info.get("calibration_path"),
            "temperature": float(self.temperature),
            "uncalibrated_prob": primary["uncalibrated_prob"],
            "calibrated_prob": primary["fake_probability"],
            "raw_logit": primary["raw_logit"],
            "decision_threshold": float(active_policy["decision_threshold"]),
            "default_threshold": 0.50,
            "uncertain_low": float(active_policy["uncertain_low"]),
            "uncertain_high": float(active_policy["uncertain_high"]),
            "operating_mode": operating_mode,
            "classifier_confidence": float(primary["outcome_res"]["classifier_confidence"]),
            "margin_from_threshold": float(primary["outcome_res"]["margin_from_threshold"]),
            "predictive_entropy_bits": float(primary["outcome_res"]["predictive_entropy_bits"]),
            "spatial_fake_prob": primary.get("spatial_fake_prob"),
            "frequency_fake_prob": primary.get("frequency_fake_prob"),
            "branch_discrepancy": primary.get("branch_discrepancy", 0.0),
            "has_branch_conflict": primary.get("has_branch_conflict", False),
            "false_negative_risk": primary.get("false_negative_risk", False),
            "bayesian_override_removed": True,
        }

        forensic_observations = self._build_forensic_observations(primary, faces_detected_count, cal_meta)

        t_fft_total = sum(f["t_fft"] for f in analyzed_faces)
        t_ela_total = sum(f["t_ela"] for f in analyzed_faces)
        t_srm_total = sum(f.get("t_srm", 0.0) for f in analyzed_faces)
        t_cam_total = sum(f["t_gradcam"] for f in analyzed_faces)
        t_prep_tensor_total = sum(f.get("t_prep_tensor", 0.0) for f in analyzed_faces)

        total_time_ms = (time.time() - start_time) * 1000.0
        t_preprocessing = t_face + t_prep_tensor_total
        t_explainability = t_fft_total + t_ela_total + t_srm_total + t_cam_total
        t_neural = t_neural_batch_ms
        t_postprocessing = max(0.0, total_time_ms - (t_preprocessing + t_explainability + t_neural))

        latency_breakdown = {
            "face_detection_ms": round(t_face, 1),
            "spectral_fft_ms": round(t_fft_total, 1),
            "ela_compression_ms": round(t_ela_total, 1),
            "srm_residual_ms": round(t_srm_total, 1),
            "neural_inference_ms": round(t_neural, 1),
            "gradcam_backward_ms": round(t_cam_total, 1),
            "preprocessing_ms": round(t_preprocessing, 1),
            "explainability_ms": round(t_explainability, 1),
            "postprocessing_ms": round(t_postprocessing, 1),
            "total_ms": round(total_time_ms, 1),
        }
        mem_telemetry = get_runtime_memory_telemetry(self.device)
        performance_telemetry = {
            "device": self.device.type,
            "cpu_fallback_triggered": bool(fallback_triggered),
            "batched_inference": True,
            "batch_size": len(prepared_faces),
            "faces_evaluated": len(prepared_faces),
            "throughput_fps": compute_throughput_fps(1, total_time_ms),
            "images_per_second": compute_throughput_fps(1, total_time_ms),
            "throughput_label": "images/second",
            "neural_throughput_fps": compute_throughput_fps(len(prepared_faces), t_neural_batch_ms),
            "neural_inference_ms": round(t_neural_batch_ms, 2),
            **mem_telemetry,
        }

        multi_face_reports = [
            {
                "face_index": f["face_index"],
                "is_primary": f["is_primary"],
                "bbox": f["bbox"],
                "raw_face_bbox": f.get("raw_face_bbox"),
                "object_bbox": f.get("object_bbox", f["bbox"]),
                "face_detected": f["face_detected"],
                "has_face": f.get("has_face", f["face_detected"]),
                "image_type": f.get("image_type", "Human Face Portrait" if f["face_detected"] else "General Object"),
                "aligned_face": f["aligned_face"],
                "fake_probability": f["fake_probability"],
                "real_probability": f["real_probability"],
                "confidence_percent": f["confidence_percent"],
                "prediction_outcome": f["prediction_outcome"],
                "is_fake": f["is_fake"],
                "is_uncertain": f["is_uncertain"],
                "uncertainty_reason": f["uncertainty_reason"],
                "verdict_tier": f["verdict_tier"],
                "verdict_badge": f["verdict_badge"],
                "gradcam_heatmap": f["gradcam_heatmap"],
                "gradcam_overlay": f["gradcam_overlay"],
                "spatial_landmarks": f["spatial_landmarks"],
                "fft_spectrum_img": f["fft_spectrum_img"],
                "fft_log_colormap_img": f["fft_log_colormap_img"],
                "fft_anomaly_map": f["fft_anomaly_map"],
                "azimuthal_frequencies": f["azimuthal_frequencies"],
                "azimuthal_profile": f["azimuthal_profile"],
                "spectral_details": f["spectral_details"],
                "spectral_comparison": f["spectral_comparison"],
                "ela_image": f["ela_image"],
                "ela_heatmap_color": f["ela_heatmap_color"],
                "ela_mean_error": f["ela_mean_error"],
                "ela_details": f["ela_details"],
                "srm_details": f.get("srm_details"),
                "spatial_fake_prob": f.get("spatial_fake_prob"),
                "spatial_logit": f.get("spatial_logit"),
                "frequency_fake_prob": f.get("frequency_fake_prob"),
                "frequency_logit": f.get("frequency_logit"),
                "branch_discrepancy": f.get("branch_discrepancy", 0.0),
                "has_branch_conflict": f.get("has_branch_conflict", False),
                "false_negative_risk": f.get("false_negative_risk", False),
                "false_negative_warning": f.get("false_negative_warning"),
                "evidence_consistency": f.get("evidence_consistency", "CONSISTENT"),
                "operating_mode": f.get("operating_mode", "balanced"),
            }
            for f in analyzed_faces
        ]

        primary_f_rec = all_face_records[0]
        has_face_flag = bool(primary_f_rec.get("has_face", primary["face_detected"]))
        image_type_label = primary_f_rec.get("image_type", "Human Face Portrait" if has_face_flag else "General Object")
        object_bbox_tuple = primary_f_rec.get("object_bbox", primary["bbox"])

        return {
            "prediction_label": primary["prediction_label"],
            "prediction_outcome": primary["prediction_outcome"],
            "is_fake": primary["is_fake"],
            "is_uncertain": primary["is_uncertain"],
            "uncertainty_reason": primary["uncertainty_reason"],
            "fake_probability": primary["fake_probability"],
            "real_probability": primary["real_probability"],
            "confidence_percent": primary["confidence_percent"],
            "verdict_tier": primary["verdict_tier"],
            "verdict_badge": primary["verdict_badge"],
            "verdict_class": primary["verdict_class"],
            "verdict_explanation": primary["verdict_explanation"],
            "faces_count": faces_detected_count,
            "full_image_annotated": full_image_annotated,
            "multi_face_reports": multi_face_reports,
            "face_detected": primary["face_detected"],
            "has_face": has_face_flag,
            "image_type": image_type_label,
            "object_bbox": object_bbox_tuple,
            "bbox": primary["bbox"],
            "aligned_face": primary["aligned_face"],
            "aligned_face_np": primary["aligned_face_np"],
            "gradcam_heatmap": primary["gradcam_heatmap"],
            "gradcam_overlay": primary["gradcam_overlay"],
            "fft_spectrum_img": primary["fft_spectrum_img"],
            "fft_log_colormap_img": primary["fft_log_colormap_img"],
            "fft_anomaly_map": primary["fft_anomaly_map"],
            "fft_anomaly_bands": primary["fft_anomaly_bands"],
            "azimuthal_frequencies": primary["azimuthal_frequencies"],
            "azimuthal_profile": primary["azimuthal_profile"],
            "spectral_details": primary["spectral_details"],
            "spectral_comparison": primary["spectral_comparison"],
            "ela_image": primary["ela_image"],
            "ela_heatmap_color": primary["ela_heatmap_color"],
            "ela_mean_error": primary["ela_mean_error"],
            "ela_details": primary["ela_details"],
            "ela_comparison": primary["ela_comparison"],
            "ela_limitations": primary["ela_limitations"],
            "srm_details": primary.get("srm_details"),
            "srm_heatmap": primary.get("srm_details", {}).get("heatmap_image"),
            "spatial_landmarks": primary["spatial_landmarks"],
            "forensic_observations": forensic_observations,
            "inference_time_ms": total_time_ms,
            "latency_breakdown": latency_breakdown,
            "performance_telemetry": performance_telemetry,
            "laplacian_variance": primary["laplacian_variance"],
            "calibration_metadata": cal_meta,
            "scientific_disclaimer": "AI forensic evaluations provide probabilistic evidence based on multi-domain statistical modeling and must be corroborated with original media provenance.",
            "forensic_evidence": {
                "neural_fake_prob": primary["fake_probability"],
                "neural_uncalibrated_prob": primary["uncalibrated_prob"],
                "neural_raw_logit": primary["raw_logit"],
                "spectral_anomaly_prob": primary["p_spectral"],
                "ela_anomaly_prob": primary["p_ela"],
                "srm_block_cv": primary.get("srm_details", {}).get("block_inconsistency_cv", 0.0),
                "srm_mean_energy": primary.get("srm_details", {}).get("mean_energy", 0.0),
                "spatial_branch_fake_prob": primary.get("spatial_fake_prob"),
                "frequency_branch_fake_prob": primary.get("frequency_fake_prob"),
                "branch_discrepancy": primary.get("branch_discrepancy", 0.0),
                "has_branch_conflict": primary.get("has_branch_conflict", False),
                "false_negative_risk": primary.get("false_negative_risk", False),
                "operating_mode": operating_mode,
                "neural_confidence": float(primary["outcome_res"]["classifier_confidence"]),
                "laplacian_sharpness": primary["laplacian_variance"],
                "evidence_decoupled_from_classifier": True,
                "heuristic_override_applied": False,
            },
            "operating_mode": operating_mode,
            "spatial_fake_prob": primary.get("spatial_fake_prob"),
            "spatial_logit": primary.get("spatial_logit"),
            "frequency_fake_prob": primary.get("frequency_fake_prob"),
            "frequency_logit": primary.get("frequency_logit"),
            "branch_discrepancy": primary.get("branch_discrepancy", 0.0),
            "has_branch_conflict": primary.get("has_branch_conflict", False),
            "false_negative_risk": primary.get("false_negative_risk", False),
            "false_negative_warning": primary.get("false_negative_warning"),
            "evidence_consistency": primary.get("evidence_consistency", "CONSISTENT"),
        }
