"""
Frame-Level Video Deepfake Detection & Timeline Forensics Module.

Supports MP4, AVI, and MOV containers via OpenCV, configurable frame sampling,
per-frame face extraction & preprocessing, batched DualDomainSentinelNet inference
using validation-calibrated temperature scaling and 3-outcome uncertainty handling,
robust multi-frame score aggregation (immune to single-frame false positives),
suspicious timestamp interval extraction, and timeline visualization metadata.

Important Methodological Scope:
  This module performs independent FRAME-LEVEL spatial and 2D FFT spectral evaluation
  across sampled frames. It does NOT implement temporal sequence modeling (e.g., 3D CNN,
  optical-flow, or inter-frame LSTM/Transformer) and explicitly declares `temporal_modeling=False`.
"""

from datetime import datetime, timezone
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    OUTCOME_MANIPULATED,
    OUTCOME_UNCERTAIN,
    classify_with_policy,
    default_uncertainty_policy,
    load_calibration_policy,
    sigmoid_np,
)
from src.explainability.spectral_analysis import SpectralAnalyzer
from src.models.dual_domain_net import DualDomainSentinelNet

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
DEFAULT_MAX_VIDEO_SIZE_MB = 250.0
FRAME_OUTCOME_NO_FACE = "NO FACE DETECTED"

OUTCOME_COLOR_MAP = {
    OUTCOME_AUTHENTIC: "green",
    OUTCOME_MANIPULATED: "red",
    OUTCOME_UNCERTAIN: "yellow",
    FRAME_OUTCOME_NO_FACE: "gray",
}

OUTCOME_HEX_MAP = {
    "green": "#10b981",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "gray": "#94a3b8",
}


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS.ms (e.g., 01:04.25)."""
    sec = max(0.0, float(seconds))
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:05.2f}"


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes streaming cryptographic SHA-256 digest of raw file bytes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resize_preview_pil(pil_img: Image.Image, max_width: int = 560) -> Image.Image:
    """Downscales large video frames safely for responsive timeline inspection without high RAM."""
    w, h = pil_img.size
    if w <= max_width:
        return pil_img.copy()
    scale = max_width / float(w)
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((max_width, new_h), Image.Resampling.BILINEAR)


def _draw_face_boxes_on_preview(
    frame_rgb: Image.Image,
    bboxes: Sequence[Tuple[int, int, int, int]],
    color_hex: str,
) -> Image.Image:
    """Renders bounding boxes for detected faces on a downscaled preview frame."""
    orig_w, orig_h = frame_rgb.size
    preview = _resize_preview_pil(frame_rgb, max_width=560)
    if not bboxes:
        return preview
    pw, ph = preview.size
    sx = pw / max(orig_w, 1)
    sy = ph / max(orig_h, 1)
    draw = ImageDraw.Draw(preview)
    for (x, y, w, h) in bboxes:
        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))
        draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=3)
    return preview


def validate_video_file(
    video_path: str | Path,
    max_size_mb: float = DEFAULT_MAX_VIDEO_SIZE_MB,
    allowed_extensions: Sequence[str] = tuple(sorted(SUPPORTED_VIDEO_EXTENSIONS)),
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validates existence, extension (.mp4/.avi/.mov), byte size, and OpenCV readability.

    Returns:
        (is_valid, error_message, metadata_dict)
    """
    p = Path(video_path)
    meta: Dict[str, Any] = {
        "filename": p.name,
        "filepath": str(p),
        "extension": p.suffix.lower(),
        "file_size_bytes": 0,
        "file_size_mb": 0.0,
        "sha256": None,
        "fps": 0.0,
        "fps_estimated": False,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "duration_sec": 0.0,
    }

    if not p.exists() or not p.is_file():
        return False, f"Video file not found: {p}", meta

    ext = p.suffix.lower()
    allowed_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed_extensions}
    if ext not in allowed_set:
        return (
            False,
            f"Unsupported video format '{ext or 'none'}'. Accepted formats: {', '.join(sorted(allowed_set))}.",
            meta,
        )

    size_bytes = int(p.stat().st_size)
    meta["file_size_bytes"] = size_bytes
    meta["file_size_mb"] = round(size_bytes / (1024.0 * 1024.0), 4)
    if size_bytes == 0:
        return False, "Video file is empty (0 bytes) and cannot be decoded.", meta

    if size_bytes > max_size_mb * 1024.0 * 1024.0:
        return (
            False,
            f"Video file size ({meta['file_size_mb']:.2f} MB) exceeds maximum limit ({max_size_mb:.1f} MB).",
            meta,
        )

    meta["sha256"] = compute_file_sha256(p)

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        return False, f"Corrupted or unreadable video container: OpenCV could not open '{p.name}'.", meta

    try:
        ok, first_frame = cap.read()
        if not ok or first_frame is None or first_frame.size == 0:
            return False, f"Corrupted video stream: failed to decode initial frame from '{p.name}'.", meta

        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_estimated = False
        if not np.isfinite(raw_fps) or raw_fps <= 0.1 or raw_fps > 240.0:
            raw_fps = 25.0
            fps_estimated = True

        raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or first_frame.shape[1])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or first_frame.shape[0])

        meta["fps"] = round(raw_fps, 4)
        meta["fps_estimated"] = fps_estimated
        meta["frame_count"] = max(raw_frame_count, 0)
        meta["width"] = width
        meta["height"] = height
        meta["duration_sec"] = round((raw_frame_count / raw_fps), 3) if raw_frame_count > 0 else 0.0
        return True, None, meta
    except Exception as exc:
        return False, f"Failed while inspecting video container '{p.name}': {exc}", meta
    finally:
        cap.release()


def _select_sample_indices(
    total_frames: int,
    fps: float,
    sample_mode: str = "fps",
    sample_fps: float = 1.0,
    sample_every_n_frames: int = 15,
    max_frames: int = 32,
) -> List[int]:
    """Computes sorted unique 0-based frame indices to decode based on the requested sampling mode."""
    if total_frames <= 0:
        return []
    cap_frames = max(1, int(max_frames))

    if sample_mode == "uniform":
        n = min(cap_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()
        return sorted(set(int(i) for i in indices))

    if sample_mode == "interval":
        step = max(1, int(sample_every_n_frames))
    else:
        # "fps" sampling: sample ~sample_fps frames per second
        eff_fps = max(0.1, float(fps))
        step = max(1, int(round(eff_fps / max(0.1, float(sample_fps)))))

    indices = list(range(0, total_frames, step))
    if not indices:
        indices = [0]

    # Always cover full timeline fairly within max_frames cap
    if len(indices) > cap_frames:
        sub_pos = np.linspace(0, len(indices) - 1, num=cap_frames, dtype=int)
        indices = [indices[k] for k in sub_pos]

    return sorted(set(int(i) for i in indices))


def extract_suspicious_intervals(
    frame_records: Sequence[Dict[str, Any]],
    sample_interval_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Groups contiguous (or gap <= 1.5 * sample_interval_sec) suspicious frames into time intervals.
    Captures both MANIPULATED and UNCERTAIN clusters transparently without altering global verdicts.
    """
    suspicious_frames = [
        r for r in frame_records
        if r.get("has_face") and r.get("prediction_outcome") in {OUTCOME_MANIPULATED, OUTCOME_UNCERTAIN}
    ]
    if not suspicious_frames:
        return []

    max_gap = max(1.5 * max(sample_interval_sec, 0.25), 0.75)
    intervals: List[List[Dict[str, Any]]] = [[suspicious_frames[0]]]

    for rec in suspicious_frames[1:]:
        prev = intervals[-1][-1]
        gap = float(rec["timestamp_sec"]) - float(prev["timestamp_sec"])
        same_severity = rec["prediction_outcome"] == prev["prediction_outcome"]
        if gap <= max_gap and same_severity:
            intervals[-1].append(rec)
        else:
            intervals.append([rec])

    summaries: List[Dict[str, Any]] = []
    for group in intervals:
        probs = [float(g["fake_probability"]) for g in group if g.get("fake_probability") is not None]
        start_s = float(group[0]["timestamp_sec"])
        end_s = float(group[-1]["timestamp_sec"])
        summaries.append(
            {
                "start_sec": round(start_s, 3),
                "end_sec": round(end_s, 3),
                "start_timestamp": format_timestamp(start_s),
                "end_timestamp": format_timestamp(end_s),
                "duration_sec": round(max(0.0, end_s - start_s), 3),
                "frame_indices": [int(g["frame_index"]) for g in group],
                "frame_count": len(group),
                "severity": group[0]["prediction_outcome"],
                "peak_fake_probability": round(max(probs), 4) if probs else None,
                "mean_fake_probability": round(float(np.mean(probs)), 4) if probs else None,
            }
        )
    return summaries


def aggregate_video_scores(
    frame_records: Sequence[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregates frame-level calibrated probabilities & 3-outcome predictions into a video-level assessment.

    Safeguards enforced:
      1. Zero-face videos NEVER receive an AUTHENTIC or MANIPULATED classification.
      2. A single suspicious/manipulated frame NEVER causes an entire video to be classified
         as MANIPULATED (requires sustained multi-frame support: >= max(2, ceil(0.30 * N_face_frames))).
      3. Reports aggregation strategy, summary statistics, and explicit frame-level scope.
    """
    evaluated = [r for r in frame_records if r.get("has_face") and r.get("fake_probability") is not None]
    n_eval = len(evaluated)

    tau = float(policy.get("decision_threshold", 0.50))
    u_low = float(policy.get("uncertain_low", max(0.05, tau - 0.10)))
    u_high = float(policy.get("uncertain_high", min(0.95, tau + 0.10)))

    aggregation_method = (
        "Frame-level 10% symmetric trimmed mean with sustained multi-frame quorum "
        "(requires >= max(2, ceil(30% of evaluated face frames)) for MANIPULATED; "
        "a single suspicious frame cannot trigger a MANIPULATED video verdict)"
    )

    if n_eval == 0:
        return {
            "status": "no_faces_detected",
            "evaluated_face_frames": 0,
            "authentic_frame_count": 0,
            "manipulated_frame_count": 0,
            "uncertain_frame_count": 0,
            "authentic_frame_pct": 0.0,
            "manipulated_frame_pct": 0.0,
            "uncertain_frame_pct": 0.0,
            "overall_assessment": "NO FACES DETECTED / UNABLE TO ASSESS",
            "prediction_outcome": FRAME_OUTCOME_NO_FACE,
            "is_fake": False,
            "is_uncertain": True,
            "verdict_class": "verdict-inconclusive",
            "verdict_badge": "⚠️ NO FACES DETECTED IN SAMPLED VIDEO FRAMES",
            "verdict_explanation": (
                "No verifiable human faces were detected across sampled video frames. "
                "Deepfake facial forensics cannot assess videos without visible face regions."
            ),
            "aggregated_fake_probability": None,
            "aggregated_confidence_percent": 0.0,
            "mean_fake_probability": None,
            "median_fake_probability": None,
            "trimmed_mean_fake_probability": None,
            "p90_fake_probability": None,
            "max_fake_probability": None,
            "min_fake_probability": None,
            "aggregation_method": aggregation_method,
            "sustained_quorum_required": 2,
            "isolated_spike_suppressed": False,
        }

    probs = np.array([float(r["fake_probability"]) for r in evaluated], dtype=np.float64)
    mean_p = float(np.mean(probs))
    median_p = float(np.median(probs))
    p90_p = float(np.percentile(probs, 90))
    max_p = float(np.max(probs))
    min_p = float(np.min(probs))

    if n_eval >= 5:
        k = max(1, int(math.floor(0.10 * n_eval)))
        sorted_p = np.sort(probs)
        trimmed = sorted_p[k: n_eval - k]
        trimmed_mean_p = float(np.mean(trimmed)) if len(trimmed) > 0 else median_p
    else:
        trimmed_mean_p = median_p

    n_auth = int(sum(1 for r in evaluated if r["prediction_outcome"] == OUTCOME_AUTHENTIC))
    n_manip = int(sum(1 for r in evaluated if r["prediction_outcome"] == OUTCOME_MANIPULATED))
    n_uncert = int(sum(1 for r in evaluated if r["prediction_outcome"] == OUTCOME_UNCERTAIN))

    auth_pct = round(100.0 * n_auth / n_eval, 2)
    manip_pct = round(100.0 * n_manip / n_eval, 2)
    uncert_pct = round(100.0 * n_uncert / n_eval, 2)

    auth_ratio = n_auth / n_eval
    manip_ratio = n_manip / n_eval

    # Quorum safeguard: NEVER allow 1 single frame to classify an entire video as MANIPULATED
    sustained_quorum = max(2, int(math.ceil(0.30 * n_eval)))
    isolated_spike_suppressed = (n_manip == 1)

    if n_eval < 2:
        # Single evaluated frame cannot support multi-frame video corroboration
        outcome = OUTCOME_UNCERTAIN
        overall_assessment = f"{OUTCOME_UNCERTAIN} (SINGLE-FRAME EVIDENCE ONLY)"
        verdict_class = "verdict-inconclusive"
        verdict_badge = "⚖️ VIDEO ASSESSMENT: UNCERTAIN / NEEDS MANUAL REVIEW"
        verdict_explanation = (
            "Only 1 frame with a detected face was available. Multi-frame corroboration is required "
            "before issuing a video-level AUTHENTIC or MANIPULATED decision."
        )
    elif n_manip >= sustained_quorum and (trimmed_mean_p >= tau or manip_ratio >= 0.45):
        outcome = OUTCOME_MANIPULATED
        overall_assessment = OUTCOME_MANIPULATED
        verdict_class = "verdict-fake"
        verdict_badge = "🚨 VIDEO ASSESSMENT: MANIPULATED (FRAME-LEVEL AGGREGATION)"
        verdict_explanation = (
            f"Sustained frame-level anomalies detected across {n_manip}/{n_eval} face frames ({manip_pct:.1f}%), "
            f"meeting the multi-frame quorum (>={sustained_quorum} frames) with trimmed mean P={trimmed_mean_p:.4f}."
        )
    elif (n_manip == 0 or (n_manip == 1 and n_eval >= 5 and manip_ratio <= 0.20)) and auth_ratio >= 0.65 and trimmed_mean_p < u_high and median_p < tau:
        outcome = OUTCOME_AUTHENTIC
        overall_assessment = OUTCOME_AUTHENTIC
        verdict_class = "verdict-real"
        verdict_badge = "✅ VIDEO ASSESSMENT: AUTHENTIC (FRAME-LEVEL AGGREGATION)"
        extra_note = (
            " Note: 1 isolated suspicious frame was flagged in the timeline for optional inspection "
            "without overriding the predominantly authentic sequence."
            if isolated_spike_suppressed
            else ""
        )
        verdict_explanation = (
            f"Consistent authentic frame predictions across {n_auth}/{n_eval} face frames ({auth_pct:.1f}%), "
            f"trimmed mean P={trimmed_mean_p:.4f} below threshold {tau:.3f}.{extra_note}"
        )
    else:
        outcome = OUTCOME_UNCERTAIN
        overall_assessment = OUTCOME_UNCERTAIN
        verdict_class = "verdict-inconclusive"
        verdict_badge = "⚖️ VIDEO ASSESSMENT: UNCERTAIN / NEEDS MANUAL REVIEW"
        if isolated_spike_suppressed and n_eval < 5:
            verdict_explanation = (
                f"Detected 1 isolated suspicious frame out of {n_eval} evaluated frames; single-frame evidence "
                "cannot classify a video as manipulated, so this clip is routed for manual review."
            )
        else:
            verdict_explanation = (
                f"Mixed or borderline frame evidence ({auth_pct:.1f}% Authentic, {manip_pct:.1f}% Manipulated, "
                f"{uncert_pct:.1f}% Uncertain; trimmed mean P={trimmed_mean_p:.4f}). Escalate to manual inspection."
            )

    agg_conf = (trimmed_mean_p if outcome == OUTCOME_MANIPULATED else (1.0 - trimmed_mean_p if outcome == OUTCOME_AUTHENTIC else max(trimmed_mean_p, 1.0 - trimmed_mean_p))) * 100.0

    return {
        "status": "completed",
        "evaluated_face_frames": n_eval,
        "authentic_frame_count": n_auth,
        "manipulated_frame_count": n_manip,
        "uncertain_frame_count": n_uncert,
        "authentic_frame_pct": auth_pct,
        "manipulated_frame_pct": manip_pct,
        "uncertain_frame_pct": uncert_pct,
        "overall_assessment": overall_assessment,
        "prediction_outcome": outcome,
        "is_fake": bool(outcome == OUTCOME_MANIPULATED),
        "is_uncertain": bool(outcome != OUTCOME_MANIPULATED and outcome != OUTCOME_AUTHENTIC),
        "verdict_class": verdict_class,
        "verdict_badge": verdict_badge,
        "verdict_explanation": verdict_explanation,
        "aggregated_fake_probability": round(trimmed_mean_p, 5),
        "aggregated_confidence_percent": round(float(agg_conf), 2),
        "mean_fake_probability": round(mean_p, 5),
        "median_fake_probability": round(median_p, 5),
        "trimmed_mean_fake_probability": round(trimmed_mean_p, 5),
        "p90_fake_probability": round(p90_p, 5),
        "max_fake_probability": round(max_p, 5),
        "min_fake_probability": round(min_p, 5),
        "aggregation_method": aggregation_method,
        "sustained_quorum_required": sustained_quorum,
        "isolated_spike_suppressed": isolated_spike_suppressed,
    }


def _compute_confidence_distribution_summary(frame_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes distribution statistics over frame-level predictions and confidences."""
    evaluated = [r for r in frame_records if r.get("has_face") and r.get("fake_probability") is not None]
    if not evaluated:
        return {
            "count": 0,
            "fake_probabilities": [],
            "confidences": [],
            "hist_counts": [],
            "hist_edges": np.linspace(0.0, 1.0, 11).tolist(),
        }

    probs = [float(r["fake_probability"]) for r in evaluated]
    confs = [float(r["classifier_confidence"]) for r in evaluated]
    counts, edges = np.histogram(probs, bins=10, range=(0.0, 1.0))
    return {
        "count": len(evaluated),
        "fake_probabilities": probs,
        "confidences": confs,
        "hist_counts": [int(c) for c in counts],
        "hist_edges": [round(float(e), 3) for e in edges],
        "prob_std": round(float(np.std(probs)), 5),
        "conf_mean": round(float(np.mean(confs)), 5),
        "conf_min": round(float(np.min(confs)), 5),
        "conf_max": round(float(np.max(confs)), 5),
    }


class VideoForensicAnalyzer:
    """
    Frame-level deepfake video forensic analyzer reusing DualDomainSentinelNet
    and validation-derived calibration/uncertainty policies.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_model.pth",
        config_path: str = "config/default_config.yaml",
        device: str = "auto",
        calibration_path: Optional[str] = "results/calibration/calibration_results.json",
        pipeline: Optional[Any] = None,
        max_file_size_mb: float = DEFAULT_MAX_VIDEO_SIZE_MB,
    ):
        self.max_file_size_mb = float(max_file_size_mb)
        self.config = load_config(config_path) if Path(config_path).exists() else {}

        if pipeline is not None:
            self.device = pipeline.device
            self.model = pipeline.model
            self.face_extractor = pipeline.face_extractor
            self.spectral_analyzer = pipeline.spectral_analyzer
            self.calibration_info = dict(getattr(pipeline, "calibration_info", {}))
            self.temperature = float(getattr(pipeline, "temperature", 1.0))
            self.uncertainty_policy = dict(
                getattr(pipeline, "uncertainty_policy", default_uncertainty_policy())
            )
            self._shared_pipeline = pipeline
        else:
            self._shared_pipeline = None
            self.device = get_device(device)
            self.face_extractor = FaceExtractor(target_size=(256, 256))
            self.spectral_analyzer = SpectralAnalyzer(target_size=256)

            model_cfg = self.config.get("model", {})
            self.model = DualDomainSentinelNet(
                backbone_name=model_cfg.get("backbone", "efficientnet_b0"),
                pretrained=False,
                spatial_dim=model_cfg.get("spatial_feat_dim", 512),
                spectral_dim=model_cfg.get("spectral_feat_dim", 256),
                fusion_type=model_cfg.get("fusion_type", "cross_attention"),
            ).to(self.device)

            if checkpoint_path and Path(checkpoint_path).exists():
                loaded = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
                self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()

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
                "min_laplacian_sharpness": float(self.calibration_info.get("min_laplacian_sharpness", 15.0)),
                "min_dimension_px": int(self.calibration_info.get("min_dimension_px", 80)),
            }

    def _decode_sampled_frames(
        self,
        video_path: Path,
        meta: Dict[str, Any],
        sample_mode: str = "fps",
        sample_fps: float = 1.0,
        sample_every_n_frames: int = 15,
        max_frames: int = 32,
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """Extracts sampled RGB frames using OpenCV, supporting variable or missing header counts."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], 0, 0.0

        fps = float(meta.get("fps") or 25.0)
        reported_count = int(meta.get("frame_count") or 0)
        decoded_frames: List[Dict[str, Any]] = []
        unreadable_count = 0
        actual_total_frames = reported_count

        try:
            if reported_count > 0:
                target_indices = _select_sample_indices(
                    total_frames=reported_count,
                    fps=fps,
                    sample_mode=sample_mode,
                    sample_fps=sample_fps,
                    sample_every_n_frames=sample_every_n_frames,
                    max_frames=max_frames,
                )
                for idx in target_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        unreadable_count += 1
                        continue
                    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    ts_sec = (pos_msec / 1000.0) if (pos_msec > 0.0 or idx == 0) else (idx / fps)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    decoded_frames.append(
                        {
                            "frame_index": int(idx),
                            "timestamp_sec": round(float(ts_sec), 3),
                            "rgb_pil": Image.fromarray(rgb),
                        }
                    )

                # If container POS seeking skipped unreadable frames, retry sequential scan once
                if not decoded_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reported_count = 0

            if reported_count <= 0:
                step = max(1, int(sample_every_n_frames)) if sample_mode == "interval" else max(1, int(round(fps / max(0.1, sample_fps))))
                idx = 0
                raw_walk: List[Dict[str, Any]] = []
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        break
                    if idx % step == 0:
                        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                        ts_sec = (pos_msec / 1000.0) if (pos_msec > 0.0 or idx == 0) else (idx / fps)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        raw_walk.append(
                            {
                                "frame_index": int(idx),
                                "timestamp_sec": round(float(ts_sec), 3),
                                "rgb_pil": Image.fromarray(rgb),
                            }
                        )
                    idx += 1
                actual_total_frames = idx
                if len(raw_walk) > max_frames:
                    keep = np.linspace(0, len(raw_walk) - 1, num=max_frames, dtype=int)
                    decoded_frames = [raw_walk[k] for k in keep]
                else:
                    decoded_frames = raw_walk
        finally:
            cap.release()

        duration_sec = meta.get("duration_sec") or 0.0
        if duration_sec <= 0.0 and actual_total_frames > 0:
            duration_sec = round(actual_total_frames / max(fps, 0.1), 3)
        elif decoded_frames and duration_sec <= 0.0:
            duration_sec = round(max(d["timestamp_sec"] for d in decoded_frames), 3)

        meta["frame_count"] = max(actual_total_frames, len(decoded_frames))
        meta["duration_sec"] = duration_sec
        return decoded_frames, unreadable_count, duration_sec

    @torch.no_grad()
    def analyze_video(
        self,
        video_path: str | Path,
        sample_mode: str = "fps",
        sample_fps: float = 1.0,
        sample_every_n_frames: int = 15,
        max_frames: int = 30,
        batch_size: int = 16,
        threshold: Optional[float] = None,
        allow_center_crop_fallback: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete frame-level forensic examination on a video file.

        Args:
            video_path: Path to MP4, AVI, or MOV video file.
            sample_mode: 'fps', 'interval', or 'uniform'.
            sample_fps: Target sampled frames per second when sample_mode='fps'.
            sample_every_n_frames: Frame stride when sample_mode='interval'.
            max_frames: Maximum number of frames decoded & evaluated.
            batch_size: Mini-batch size for batched DualDomainSentinelNet forward passes.
            threshold: Optional threshold override (defaults to calibrated policy threshold).
            allow_center_crop_fallback: If True, evaluates center-crop even if Haar finds no face.
            progress_callback: Optional `(fraction_0_to_1, status_text)` callback for UI.
        """
        t0 = time.time()
        p = Path(video_path)

        if progress_callback:
            progress_callback(0.05, "Validating video container & cryptographic fingerprint...")

        is_valid, err_msg, meta = validate_video_file(p, max_size_mb=self.max_file_size_mb)
        if not is_valid:
            return {
                "status": "error",
                "error": err_msg,
                "filename": meta["filename"],
                "file_metadata": meta,
                "duration_sec": 0.0,
                "frame_count": 0,
                "processed_frames": 0,
                "detected_faces": 0,
                "frames_with_faces": 0,
                "frames_without_faces": 0,
                "authentic_frame_pct": 0.0,
                "manipulated_frame_pct": 0.0,
                "uncertain_frame_pct": 0.0,
                "overall_assessment": "ERROR / INVALID OR CORRUPTED VIDEO",
                "prediction_outcome": "ERROR",
                "verdict_class": "verdict-inconclusive",
                "verdict_badge": f"❌ VIDEO VALIDATION FAILED: {err_msg}",
                "verdict_explanation": err_msg,
                "suspicious_timestamps": [],
                "timeline": [],
                "confidence_distribution": _compute_confidence_distribution_summary([]),
                "analysis_scope": "frame_level_inference",
                "temporal_modeling": False,
                "methodology_disclaimer": (
                    "Frame-level spatial-spectral evaluation only. No temporal motion or inter-frame deep learning model is used."
                ),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 2),
            }

        if progress_callback:
            progress_callback(0.20, "Decoding sampled frames via OpenCV...")

        sampled_frames, unreadable_frames, duration_sec = self._decode_sampled_frames(
            video_path=p,
            meta=meta,
            sample_mode=sample_mode,
            sample_fps=sample_fps,
            sample_every_n_frames=sample_every_n_frames,
            max_frames=max_frames,
        )

        if not sampled_frames:
            err = f"Corrupted video stream: zero readable frames could be decoded from '{p.name}'."
            return {
                "status": "error",
                "error": err,
                "filename": meta["filename"],
                "file_metadata": meta,
                "duration_sec": float(duration_sec),
                "frame_count": int(meta["frame_count"]),
                "processed_frames": 0,
                "detected_faces": 0,
                "frames_with_faces": 0,
                "frames_without_faces": 0,
                "authentic_frame_pct": 0.0,
                "manipulated_frame_pct": 0.0,
                "uncertain_frame_pct": 0.0,
                "overall_assessment": "ERROR / CORRUPTED VIDEO STREAM",
                "prediction_outcome": "ERROR",
                "verdict_class": "verdict-inconclusive",
                "verdict_badge": "❌ CORRUPTED OR UNREADABLE VIDEO STREAM",
                "verdict_explanation": err,
                "suspicious_timestamps": [],
                "timeline": [],
                "confidence_distribution": _compute_confidence_distribution_summary([]),
                "analysis_scope": "frame_level_inference",
                "temporal_modeling": False,
                "methodology_disclaimer": (
                    "Frame-level spatial-spectral evaluation only. No temporal motion or inter-frame deep learning model is used."
                ),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 2),
            }

        # Prepare active 3-outcome policy
        active_policy = dict(self.uncertainty_policy)
        if threshold is not None:
            t_req = float(np.clip(threshold, 0.05, 0.95))
            lower_m = max(0.01, active_policy["decision_threshold"] - active_policy["uncertain_low"])
            upper_m = max(0.01, active_policy["uncertain_high"] - active_policy["decision_threshold"])
            active_policy["decision_threshold"] = t_req
            active_policy["uncertain_low"] = max(0.01, round(t_req - lower_m, 4))
            active_policy["uncertain_high"] = min(0.99, round(t_req + upper_m, 4))

        min_lap = float(active_policy.get("min_laplacian_sharpness", 15.0))
        min_dim = int(active_policy.get("min_dimension_px", 80))

        # Stage 1: Face detection & preprocessing across sampled frames
        if progress_callback:
            progress_callback(0.35, f"Detecting & preprocessing faces across {len(sampled_frames)} frames...")

        frame_records: List[Dict[str, Any]] = []
        face_batch_meta: List[Tuple[int, int, Dict[str, Any], float]] = []
        spatial_tensors: List[torch.Tensor] = []
        spectral_tensors: List[torch.Tensor] = []
        total_detected_faces = 0

        for f_pos, item in enumerate(sampled_frames):
            f_idx = int(item["frame_index"])
            ts_sec = float(item["timestamp_sec"])
            rgb_pil = item["rgb_pil"]

            all_recs = self.face_extractor.extract_all_faces(rgb_pil)
            valid_faces = [r for r in all_recs if r.get("face_detected")]
            if not valid_faces and allow_center_crop_fallback and all_recs:
                valid_faces = [all_recs[0]]

            total_detected_faces += len(valid_faces)
            rec_entry: Dict[str, Any] = {
                "sample_position": f_pos,
                "frame_index": f_idx,
                "timestamp_sec": ts_sec,
                "timestamp_label": format_timestamp(ts_sec),
                "has_face": len(valid_faces) > 0,
                "detected_face_count": len(valid_faces),
                "bboxes": [r["bbox"] for r in valid_faces],
                "primary_bbox": valid_faces[0]["bbox"] if valid_faces else None,
                "aligned_face": valid_faces[0]["aligned_face"] if valid_faces else None,
                "raw_preview": rgb_pil,
                "face_predictions": [],
                "fake_probability": None,
                "uncalibrated_probability": None,
                "classifier_confidence": None,
                "prediction_outcome": FRAME_OUTCOME_NO_FACE,
                "timeline_color": OUTCOME_COLOR_MAP[FRAME_OUTCOME_NO_FACE],
                "timeline_hex": OUTCOME_HEX_MAP[OUTCOME_COLOR_MAP[FRAME_OUTCOME_NO_FACE]],
                "uncertainty_reason": "No human face detected in frame." if not valid_faces else None,
            }
            frame_records.append(rec_entry)

            for face_idx, f_rec in enumerate(valid_faces):
                crop_pil = f_rec["aligned_face"]
                crop_np = np.array(crop_pil.convert("RGB"))
                gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                mag_spec, _ = self.spectral_analyzer.compute_fft_magnitude(crop_pil)
                spatial_tensors.append(SPATIAL_TRANSFORM(crop_pil))
                spectral_tensors.append(SPECTRAL_TRANSFORM(Image.fromarray(mag_spec)))
                face_batch_meta.append((f_pos, face_idx, f_rec, lap_var))

        # Stage 2: Batched DualDomainSentinelNet inference over all detected faces
        fallback_triggered = False
        neural_inference_ms = 0.0
        eff_batch_size = max(1, int(batch_size))
        if spatial_tensors:
            if progress_callback:
                progress_callback(0.65, f"Running batched DualDomainSentinelNet inference on {len(spatial_tensors)} faces...")

            self.model.eval()
            t_neural_start = time.time()
            logits_all: List[float] = []
            for start in range(0, len(spatial_tensors), eff_batch_size):
                end = min(start + eff_batch_size, len(spatial_tensors))
                sp_b = torch.stack(spatial_tensors[start:end])
                fr_b = torch.stack(spectral_tensors[start:end])
                logits_b, active_dev, did_fallback = safe_batched_forward(
                    self.model,
                    sp_b,
                    fr_b,
                    self.device,
                    enable_cpu_fallback=True,
                )
                self.device = active_dev
                if self._shared_pipeline is not None:
                    self._shared_pipeline.device = active_dev
                if did_fallback:
                    fallback_triggered = True
                logits_all.extend(logits_b.reshape(-1).detach().cpu().numpy().tolist())
            neural_inference_ms = (time.time() - t_neural_start) * 1000.0

            for (f_pos, face_idx, f_rec, lap_var), raw_logit in zip(face_batch_meta, logits_all):
                p_uncal = float(sigmoid_np(np.array([raw_logit]))[0])
                p_cal = float(sigmoid_np(np.array([raw_logit / max(self.temperature, 1e-6)]))[0])
                bbox = f_rec["bbox"]
                q_ok = (lap_var >= min_lap) and (min(bbox[2], bbox[3]) >= min(min_dim, 50))
                q_reason = None if q_ok else f"Face crop sharpness ({lap_var:.1f}) or size ({bbox[2]}x{bbox[3]}) below threshold"
                out_res = classify_with_policy(
                    calibrated_prob=p_cal,
                    policy=active_policy,
                    quality_ok=q_ok,
                    quality_reason=q_reason,
                )
                frame_records[f_pos]["face_predictions"].append(
                    {
                        "face_index": face_idx,
                        "bbox": bbox,
                        "aligned_face": f_rec["aligned_face"],
                        "raw_logit": float(raw_logit),
                        "uncalibrated_probability": p_uncal,
                        "fake_probability": p_cal,
                        "classifier_confidence": float(out_res["classifier_confidence"]),
                        "prediction_outcome": out_res["prediction_outcome"],
                        "uncertainty_reason": out_res.get("uncertainty_reason"),
                        "laplacian_sharpness": round(lap_var, 2),
                    }
                )

        # Stage 3: Finalize per-frame outcome + bounding-box preview thumbnails
        if progress_callback:
            progress_callback(0.85, "Aggregating frame predictions & identifying suspicious intervals...")

        for rec in frame_records:
            raw_pil = rec.pop("raw_preview")
            if rec["face_predictions"]:
                # Highest-risk face in the frame drives the frame inspection view
                lead_face = max(rec["face_predictions"], key=lambda x: x["fake_probability"])
                outcome = lead_face["prediction_outcome"]
                color_name = OUTCOME_COLOR_MAP.get(outcome, "yellow")
                color_hex = OUTCOME_HEX_MAP[color_name]

                rec["fake_probability"] = round(float(lead_face["fake_probability"]), 5)
                rec["uncalibrated_probability"] = round(float(lead_face["uncalibrated_probability"]), 5)
                rec["classifier_confidence"] = round(float(lead_face["classifier_confidence"]), 5)
                rec["prediction_outcome"] = outcome
                rec["uncertainty_reason"] = lead_face.get("uncertainty_reason")
                rec["primary_bbox"] = lead_face["bbox"]
                rec["aligned_face"] = lead_face["aligned_face"]
                rec["timeline_color"] = color_name
                rec["timeline_hex"] = color_hex
                rec["frame_preview"] = _draw_face_boxes_on_preview(raw_pil, rec["bboxes"], color_hex)
            else:
                rec["frame_preview"] = _draw_face_boxes_on_preview(raw_pil, [], OUTCOME_HEX_MAP["gray"])

        # Effective inter-sample gap in seconds
        if len(frame_records) >= 2:
            diffs = [
                frame_records[i + 1]["timestamp_sec"] - frame_records[i]["timestamp_sec"]
                for i in range(len(frame_records) - 1)
                if frame_records[i + 1]["timestamp_sec"] >= frame_records[i]["timestamp_sec"]
            ]
            eff_sample_interval = float(np.median(diffs)) if diffs else 1.0
        else:
            eff_sample_interval = 1.0

        agg = aggregate_video_scores(frame_records, active_policy)
        suspicious_intervals = extract_suspicious_intervals(frame_records, sample_interval_sec=eff_sample_interval)
        conf_dist = _compute_confidence_distribution_summary(frame_records)

        frames_with_faces = sum(1 for r in frame_records if r["has_face"])
        frames_without_faces = len(frame_records) - frames_with_faces
        total_ms = round((time.time() - t0) * 1000.0, 2)
        fps_proc = round((len(frame_records) / max(total_ms / 1000.0, 1e-4)), 2)

        if progress_callback:
            progress_callback(1.0, "Video forensic examination complete.")

        return {
            "status": agg["status"],
            "error": None,
            "filename": meta["filename"],
            "file_metadata": meta,
            "sha256": meta["sha256"],
            "duration_sec": float(duration_sec),
            "duration_formatted": format_timestamp(duration_sec),
            "fps": float(meta["fps"]),
            "frame_count": int(meta["frame_count"]),
            "processed_frames": len(frame_records),
            "unreadable_frames_skipped": unreadable_frames,
            "frames_with_faces": frames_with_faces,
            "frames_without_faces": frames_without_faces,
            "detected_faces": total_detected_faces,
            "authentic_frame_count": agg["authentic_frame_count"],
            "manipulated_frame_count": agg["manipulated_frame_count"],
            "uncertain_frame_count": agg["uncertain_frame_count"],
            "authentic_frame_pct": agg["authentic_frame_pct"],
            "manipulated_frame_pct": agg["manipulated_frame_pct"],
            "uncertain_frame_pct": agg["uncertain_frame_pct"],
            "overall_assessment": agg["overall_assessment"],
            "prediction_outcome": agg["prediction_outcome"],
            "is_fake": agg["is_fake"],
            "is_uncertain": agg["is_uncertain"],
            "verdict_class": agg["verdict_class"],
            "verdict_badge": agg["verdict_badge"],
            "verdict_explanation": agg["verdict_explanation"],
            "aggregated_fake_probability": agg["aggregated_fake_probability"],
            "aggregated_confidence_percent": agg["aggregated_confidence_percent"],
            "mean_fake_probability": agg["mean_fake_probability"],
            "median_fake_probability": agg["median_fake_probability"],
            "trimmed_mean_fake_probability": agg["trimmed_mean_fake_probability"],
            "p90_fake_probability": agg["p90_fake_probability"],
            "max_fake_probability": agg["max_fake_probability"],
            "min_fake_probability": agg["min_fake_probability"],
            "aggregation_method": agg["aggregation_method"],
            "sustained_quorum_required": agg["sustained_quorum_required"],
            "isolated_spike_suppressed": agg["isolated_spike_suppressed"],
            "suspicious_timestamps": suspicious_intervals,
            "timeline": frame_records,
            "confidence_distribution": conf_dist,
            "sampling_config": {
                "sample_mode": sample_mode,
                "sample_fps": float(sample_fps),
                "sample_every_n_frames": int(sample_every_n_frames),
                "max_frames": int(max_frames),
                "effective_interval_sec": round(eff_sample_interval, 3),
                "allow_center_crop_fallback": bool(allow_center_crop_fallback),
            },
            "calibration_metadata": {
                "calibrated": bool(self.calibration_info.get("calibrated", False)),
                "temperature": float(self.temperature),
                "decision_threshold": float(active_policy["decision_threshold"]),
                "uncertain_low": float(active_policy["uncertain_low"]),
                "uncertain_high": float(active_policy["uncertain_high"]),
            },
            "analysis_scope": "frame_level_inference",
            "temporal_modeling": False,
            "methodology_disclaimer": (
                "Initial video analysis applies DualDomainSentinelNet independently per sampled frame "
                "(frame-level spatial & 2D FFT evaluation). No temporal deepfake model (3D CNN, optical flow, "
                "or inter-frame transformer) is active; inter-frame temporal dynamics are not claimed."
            ),
            "inference_time_ms": total_ms,
            "processing_fps": fps_proc,
            "performance_telemetry": {
                "device": self.device.type,
                "cpu_fallback_triggered": bool(fallback_triggered),
                "batched_inference": True,
                "batch_size": eff_batch_size,
                "faces_evaluated": len(spatial_tensors),
                "neural_inference_ms": round(neural_inference_ms, 2),
                "processing_fps": fps_proc,
                "neural_throughput_fps": compute_throughput_fps(len(spatial_tensors), neural_inference_ms),
                **get_runtime_memory_telemetry(self.device),
            },
        }


def build_video_forensic_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs a strictly JSON-serializable structured video forensic report from
    `VideoForensicAnalyzer.analyze_video(...)` output (strips in-memory PIL images).
    """
    timeline_clean: List[Dict[str, Any]] = []
    for rec in results.get("timeline", []):
        timeline_clean.append(
            {
                "frame_index": int(rec.get("frame_index", 0)),
                "timestamp_sec": float(rec.get("timestamp_sec", 0.0)),
                "timestamp_label": rec.get("timestamp_label", "00:00.00"),
                "has_face": bool(rec.get("has_face", False)),
                "detected_face_count": int(rec.get("detected_face_count", 0)),
                "bboxes": [list(b) for b in rec.get("bboxes", [])],
                "fake_probability": rec.get("fake_probability"),
                "uncalibrated_probability": rec.get("uncalibrated_probability"),
                "classifier_confidence": rec.get("classifier_confidence"),
                "prediction_outcome": rec.get("prediction_outcome"),
                "timeline_color": rec.get("timeline_color"),
                "uncertainty_reason": rec.get("uncertainty_reason"),
            }
        )

    return {
        "report_type": "Deepfake Forensic Sentinel - Frame-Level Video Forensic Report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": results.get("status", "unknown"),
        "error": results.get("error"),
        "filename": results.get("filename"),
        "sha256": results.get("sha256"),
        "file_metadata": results.get("file_metadata", {}),
        "video_summary": {
            "duration_sec": results.get("duration_sec", 0.0),
            "duration_formatted": results.get("duration_formatted", "00:00.00"),
            "fps": results.get("fps", 0.0),
            "frame_count": results.get("frame_count", 0),
            "processed_frames": results.get("processed_frames", 0),
            "frames_with_faces": results.get("frames_with_faces", 0),
            "frames_without_faces": results.get("frames_without_faces", 0),
            "detected_faces": results.get("detected_faces", 0),
            "unreadable_frames_skipped": results.get("unreadable_frames_skipped", 0),
        },
        "frame_outcome_breakdown": {
            "authentic_frame_count": results.get("authentic_frame_count", 0),
            "manipulated_frame_count": results.get("manipulated_frame_count", 0),
            "uncertain_frame_count": results.get("uncertain_frame_count", 0),
            "authentic_frame_pct": results.get("authentic_frame_pct", 0.0),
            "manipulated_frame_pct": results.get("manipulated_frame_pct", 0.0),
            "uncertain_frame_pct": results.get("uncertain_frame_pct", 0.0),
        },
        "assessment": {
            "overall_assessment": results.get("overall_assessment"),
            "prediction_outcome": results.get("prediction_outcome"),
            "is_fake": results.get("is_fake", False),
            "is_uncertain": results.get("is_uncertain", True),
            "verdict_explanation": results.get("verdict_explanation"),
            "aggregated_fake_probability": results.get("aggregated_fake_probability"),
            "aggregated_confidence_percent": results.get("aggregated_confidence_percent", 0.0),
            "probability_statistics": {
                "mean": results.get("mean_fake_probability"),
                "median": results.get("median_fake_probability"),
                "trimmed_mean": results.get("trimmed_mean_fake_probability"),
                "p90": results.get("p90_fake_probability"),
                "max": results.get("max_fake_probability"),
                "min": results.get("min_fake_probability"),
            },
            "aggregation_method": results.get("aggregation_method"),
            "sustained_quorum_required": results.get("sustained_quorum_required"),
            "isolated_spike_suppressed": results.get("isolated_spike_suppressed", False),
        },
        "suspicious_timestamps": results.get("suspicious_timestamps", []),
        "confidence_distribution": results.get("confidence_distribution", {}),
        "sampling_config": results.get("sampling_config", {}),
        "calibration_metadata": results.get("calibration_metadata", {}),
        "methodology_scope": {
            "analysis_scope": results.get("analysis_scope", "frame_level_inference"),
            "temporal_modeling": bool(results.get("temporal_modeling", False)),
            "disclaimer": results.get("methodology_disclaimer", ""),
        },
        "telemetry": {
            "inference_time_ms": results.get("inference_time_ms", 0.0),
            "processing_fps": results.get("processing_fps", 0.0),
            "performance_telemetry": results.get("performance_telemetry", {}),
        },
        "timeline": timeline_clean,
    }


def format_video_forensic_markdown(report: Dict[str, Any]) -> str:
    """Formats a structured video forensic report dictionary into a clean Markdown document."""
    vs = report.get("video_summary", {})
    fb = report.get("frame_outcome_breakdown", {})
    ass = report.get("assessment", {})
    cal = report.get("calibration_metadata", {})
    scope = report.get("methodology_scope", {})
    lines = [
        f"# {report.get('report_type', 'Video Forensic Report')}",
        f"- **Generated (UTC):** `{report.get('generated_at_utc', 'N/A')}`",
        f"- **Filename:** `{report.get('filename', 'N/A')}`",
        f"- **SHA-256 Digest:** `{report.get('sha256', 'N/A')}`",
        "",
        "## 1. Overall Assessment",
        f"- **Overall Video Assessment:** **{ass.get('overall_assessment', 'N/A')}**",
        f"- **Outcome Category:** `{ass.get('prediction_outcome', 'N/A')}`",
        f"- **Aggregated Calibrated P(Manipulated):** `{ass.get('aggregated_fake_probability', 'N/A')}`",
        f"- **Aggregated Confidence:** `{ass.get('aggregated_confidence_percent', 0.0):.2f}%`",
        f"- **Explanation:** {ass.get('verdict_explanation', '')}",
        f"- **Aggregation Method:** {ass.get('aggregation_method', '')}",
        "",
        "## 2. Video & Sampling Metrics",
        f"- **Duration:** `{vs.get('duration_formatted', '00:00.00')}` ({vs.get('duration_sec', 0.0):.3f}s)",
        f"- **Container FPS:** `{vs.get('fps', 0.0)}` | **Total Frame Count:** `{vs.get('frame_count', 0)}`",
        f"- **Processed Frames:** `{vs.get('processed_frames', 0)}` | **Detected Faces:** `{vs.get('detected_faces', 0)}` across `{vs.get('frames_with_faces', 0)}` frames",
        f"- **Authentic Frames:** `{fb.get('authentic_frame_count', 0)}` ({fb.get('authentic_frame_pct', 0.0):.2f}%)",
        f"- **Manipulated Frames:** `{fb.get('manipulated_frame_count', 0)}` ({fb.get('manipulated_frame_pct', 0.0):.2f}%)",
        f"- **Uncertain Frames:** `{fb.get('uncertain_frame_count', 0)}` ({fb.get('uncertain_frame_pct', 0.0):.2f}%)",
        f"- **Calibration Policy:** `T={cal.get('temperature', 1.0):.4f}`, `tau={cal.get('decision_threshold', 0.50):.2f}`, band `[{cal.get('uncertain_low', 0.40):.3f}, {cal.get('uncertain_high', 0.60):.3f}]`",
        "",
        "## 3. Suspicious Timestamps",
    ]
    susp = report.get("suspicious_timestamps", [])
    if susp:
        lines.append("| Start | End | Duration (s) | Frames | Severity | Peak P(Fake) | Mean P(Fake) |")
        lines.append("| :--- | :--- | :---: | :---: | :--- | :---: | :---: |")
        for iv in susp:
            lines.append(
                f"| `{iv.get('start_timestamp')}` | `{iv.get('end_timestamp')}` | "
                f"{iv.get('duration_sec', 0.0):.2f} | {iv.get('frame_count', 0)} | "
                f"{iv.get('severity')} | {iv.get('peak_fake_probability', 0.0):.4f} | "
                f"{iv.get('mean_fake_probability', 0.0):.4f} |"
            )
    else:
        lines.append("No suspicious frame intervals flagged.")

    lines.extend(
        [
            "",
            "## 4. Methodological Scope & Limitations",
            f"- **Analysis Scope:** `{scope.get('analysis_scope', 'frame_level_inference')}` (`temporal_modeling={scope.get('temporal_modeling', False)}`)",
            f"- {scope.get('disclaimer', '')}",
        ]
    )
    return "\n".join(lines)
