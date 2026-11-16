# Viva Voce Defense Examination Guide (30 Expert Q&As)

**Project:** Deepfake Forensic Sentinel (`DualDomainSentinel-v1.0`)  
**Candidate:** Antriksh Mishra (Final Year B.Tech CSE Capstone)  
**Note:** Every number, file path, equation, and limitation cited below is strictly grounded in verified repository code and evaluation JSON artifacts.

---

## Category 1: Problem Motivation & Dual-Domain Foundations

### Q1. What core failure mode of conventional deepfake detectors does your project address?
**Answer:** Single-stream RGB CNNs (such as standalone EfficientNet or Xception) often overfit to donor identity, lighting, or dataset color statistics rather than physical generation artifacts. In our controlled group-disjoint ablation (`results/ablation_results.json`), the **Spatial-Only** baseline achieved `59.05%` accuracy with a `77.37%` false positive rate across unseen identity groups, whereas integrating the **2D FFT + SRM Frequency Stream** raised F1 from `0.700` to `0.822` and cut Equal Error Rate from `25.51%` to `19.55%`.

### Q2. Why does the 2D Fast Fourier Transform expose generative upsampling artifacts?
**Answer:** GANs and Diffusion decoders reconstruct `256x256` images from compact latent feature maps via transposed convolutions, pixel-shuffle layers, or strided upsampling. Whenever kernel width and stride create uneven pixel overlap, periodic spatial grid modulation emerges. The centered 2D DFT $S_{\log}(u,v) = \ln(1 + |F_{\text{shift}}(u,v)|)$ concentrates weak spatial periodicity across the entire face into localized mid/high-frequency spectral peaks and elevated radial tails.

### Q3. How does your Azimuthal Radial Profile measure spectral anomalies mathematically?
**Answer:** `SpectralAnalyzer.azimuthal_average()` bins `raw_log_spectrum` concentrically around the DC center $(u_0, v_0) = (128, 128)$ at integer radius $r = \lfloor\sqrt{(u-u_0)^2+(v-v_0)^2}\rfloor$ (`np.bincount`), producing 1D radial mean $P(r)$. We compare $P(r)$ against both a calibrated optical log-decay baseline $\hat{P}(r) = \max(0, P(1) + 1.69\ln 2 - 1.69\ln(1+r))$ and empirical local reference curves computed from authentic vs suspicious demo portraits.

### Q4. What are Spatial Rich Model (SRM) filters, and why are they fixed rather than learned?
**Answer:** Originating in digital image steganalysis, SRM filters compute high-pass directional and Laplacian finite-difference noise residuals. In `src/models/frequency_stream.py`, we register **3 fixed `5x5` kernels** (`W1 / 4`, `W2 / 12`, `W3 / 4`) as non-trainable PyTorch buffers inside `SRMConv`. Fixing them prevents the first layer from collapsing back into semantic texture matching, concatenating `1` log-spectrum channel with `3` SRM residual maps into a `4`-channel input (`4 -> 32 -> 64 -> 128 -> 256`).

### Q5. Why did Frequency-Only (Exp 2) achieve higher ROC-AUC (`0.9092`) than Full Cross-Attention (Exp 5, `0.8752`) in the ablation study?
**Answer:** On our controlled `15`-group synthetic portrait dataset (`1,426/488/486`), the generator variations inject distinct frequency-domain residuals while RGB skin tones and lighting vary across disjoint donor identities. Thus the compact `493K`-parameter Frequency Stream avoided spatial identity overfitting and reached `0.9092` ROC-AUC / `80.04%` accuracy. Meanwhile, Cross-Attention with `LabelSmoothingBCELoss(0.06)` (Exp 5) retained stronger recall (`93.0%`), achieved the **highest overall F1 (`0.822`)**, and recorded the **lowest EER (`19.55%`)** while preserving spatial Grad-CAM localization.

---

## Category 2: Neural Architecture, Cross-Attention & Training

### Q6. Walk through the exact layer dimensions and parameter count of `DualDomainSentinelNet`.
**Answer:** Our production `cross_attention` model totals **`5,882,013` parameters** (`checkpoints/best_model.pth`, SHA-256 `e8c3f62...`):
1. `SpatialBackbone.features` (EfficientNet-B0 trunk): `4,007,548` params (`0` trainable during `train.py` frozen feature caching);
2. `SpatialBackbone.proj` (`1280 -> 512` Linear + LayerNorm + ReLU): `656,896` params;
3. `FrequencyStream` (`3` fixed SRM buffers + 4-stage ConvNet + `256 -> 256` FC): `455,968` params;
4. `CrossAttentionFusion` (linear projections to `256`, 4-head `MultiheadAttention`, dual `LayerNorm`, GELU MLP): `724,224` params;
5. `classifier` MLP (`256 -> 128 -> 32 -> 1` raw logit): `37,377` params.
Total trainable head/frequency/fusion parameters under frozen-trunk training = **`1,874,465`**.

### Q7. How does `CrossAttentionFusion` outperform simple vector concatenation?
**Answer:** In `CrossAttentionFusion` (`src/models/dual_domain_net.py`), spatial embeddings project to Query $Q$ and spectral embeddings project to Key $K$ and Value $V$ across `4` heads (`d_k = 64`), followed by residual `LayerNorm` and a GELU feed-forward network. Unlike static `torch.cat` (Exp 3: `38.33 ms/img` latency, `21.81%` EER), cross-attention lets ambiguous spatial cues query spectral confidence dynamically and achieved faster single-image execution (`10.11 ms` in Exp 5) and lower EER (`19.55%`).

### Q8. Which loss function actually trained `checkpoints/best_model.pth`, and why does `loss.py` also define `BinaryFocalLoss`?
**Answer:** `src/train.py` instantiates **`LabelSmoothingBCELoss(smoothing=0.06)`**, mapping hard targets $\{0, 1\} \to [0.06, 0.94]$ to curb overconfident logits on unseen identities. `src/models/loss.py` additionally implements `BinaryFocalLoss(alpha=0.50, gamma=2.0)` for hard-example focusing if training under class imbalance. Older docs erroneously claimed Focal Loss was active in production; we audited and corrected that discrepancy (**B7/B8**).

### Q9. Why does `src/train.py` precompute `spatial_flat` features, and what is its trade-off?
**Answer:** Running `81` frozen EfficientNet-B0 layers across epochs slows training considerably. Caching `(B, 1280)` pooled vectors once reduces per-epoch training to sub-minute runs on Apple Silicon MPS. However, the trade-off (disclosed in `TECHNICAL_LIMITATIONS.md`) is that precomputing fixed spatial vectors prevents per-epoch `RandomHorizontalFlip`/`ColorJitter` augmentation on the RGB branch and locks out backbone fine-tuning unless bypassed.

---

## Category 3: Dataset Leakage, Auditing & External Benchmarks

### Q10. What severe dataset leakage did Phase 1 uncover, and how was it fixed?
**Answer:** Cryptographic hashing in Phase 1 revealed that all `2,400` images originally reused the same `5` donor portraits across train/val/test, **plus 14 exact cross-split SHA-256 duplicate groups (38 files)**—including `deepfake_stylegan_checkerboard.jpg` appearing mislabelled inside `val/real` and `test/real`! We rebuilt the benchmark with `source_split_mode="disjoint"` (`15` explicit groups: `9` train, `3` val, `3` test) and verified via `src/data/dataset_audit.py` that `results/dataset_audit.json` now has **`0` exact duplicates, `0` cross-split near duplicates, and `0` leaked source groups**.

### Q11. How do you detect near duplicates without external libraries like `imagehash`?
**Answer:** `compute_phash()` in `src/data/dataset_audit.py` computes a 64-bit 2D DCT perceptual hash natively with Pillow and `scipy.fft.dct`: resize to `32x32` grayscale -> apply 2D orthonormal DCT -> extract top-left `8x8` low-frequency coefficients -> binarize against the AC median -> compare pairs via vectorized Hamming distance (`threshold <= 4`). Our `2,400` images show `1,653` within-split augmented near duplicates and **`0` cross-split near duplicates** (median cross-split Hamming distance `16.0` bits).

### Q12. How do you enforce group-disjoint splits cleanly for ablation experiments?
**Answer:** `dataset_audit.py` uses `sklearn.model_selection.StratifiedGroupKFold` over `source_groups.json` to export `results/group_split_manifest.json` (`1,426` train / `488` val / `486` test), ensuring every synthetic variation sharing a base identity lands in strictly one split across all 5 ablation runs.

### Q13. What is the status of FaceForensics++, Celeb-DF, and DFDC evaluation?
**Answer:** `src/data/external_datasets.py` and `src/evaluation/external_benchmark.py` implement independent zero-tuning loaders (`model.eval()`, fixed threshold, uniform video clip averaging) for all three datasets. Because those restricted/large external datasets are not downloaded locally (`0/3` present), `results/external_benchmark/external_benchmark_summary.json` honestly records `available: false` and `metrics: null` rather than fabricating cross-dataset numbers. Loader behavior is fully verified with synthetic directory structures in `tests/test_external_benchmark.py`.

---

## Category 4: Calibration, Thresholding & Uncertainty Estimation

### Q14. Why did you remove the old heuristic score fusion (`0.65 Neural + 0.18 Spectral + 0.17 ELA`)?
**Answer:** Combining uncalibrated neural scores with rule-based FFT/ELA numbers using arbitrary fixed weights had zero statistical validation and caused inconsistent primary vs secondary face verdicts (**B2**). In Phase 5 we eliminated heuristic overrides completely (`heuristic_override_applied = False`). Auxiliary FFT and ELA signals remain visible as decoupled explanatory evidence, while decisions rely solely on calibrated neural probabilities.

### Q15. How does Temperature Scaling work, and why didn't it change your ROC-AUC?
**Answer:** Temperature Scaling divides raw logits by a positive scalar $T > 0$ prior to sigmoid: $\hat{p}_{\text{cal}} = \sigma(z / T)$. Because dividing by a constant $T > 0$ is strictly monotonic, sample rank order never changes, so **ROC-AUC (`0.7694` val, `0.7396` test) and PR-AUC remain invariant**. Fitting $T = 2.654271$ via CPU `L-BFGS` strictly on validation logits shrinks overconfident probabilities and cuts **validation NLL (`0.7728 -> 0.6097`)** and **held-out test NLL (`0.6247 -> 0.5571`), Brier (`0.2011 -> 0.1850`), and positive-class ECE (`0.1867 -> 0.0983`)**.

### Q16. How were $\tau = 0.71$ and the uncertainty band $[0.7070, 0.7188]$ selected without test-set leakage?
**Answer:** Both were selected exclusively on `data/val` (`N=400`, `zero_test_set_tuning = true`). Sweeping thresholds on validation selected $\tau^* = 0.71$ via **Youden's J** ($\max_\tau(\text{TPR} + \text{Specificity} - 1)$), matching best validation F1 (`0.7673`) and raising validation accuracy from `69.00%` to `77.25%`. Then `fit_uncertainty_policy_on_validation()` searched candidate bounds around `0.71` optimizing `selective_accuracy + 0.12 * coverage` subject to non-degenerate class support, choosing `[0.7070, 0.7188]` (`82.69%` selective accuracy at `83.75%` coverage on validation).

### Q17. Why did calibrated test accuracy drop from `75.50%` (`tau=0.50`) to `69.25%` (`tau=0.71`)?
**Answer:** Validation and held-out test come from disjoint portrait identity pools. Test logit scores shift slightly lower than validation calibration bins, so the validation-derived `tau = 0.71` boundary becomes ultra-conservative on test: **False Positives drop to `0` (`100%` specificity, `100%` precision)** while binary recall drops from `52.0%` (`0.50`) to `38.5%` (`0.71`). Reporting both out-of-sample behaviors transparently demonstrates genuine calibration-vs-operating-point trade-offs.

### Q18. What triggers `UNCERTAIN / NEEDS MANUAL REVIEW` besides probability falling inside `[0.7070, 0.7188]`?
**Answer:** Image quality abstention guards in `classify_with_policy()` route any crop with **Laplacian edge variance `< 15.0`** (severe blur/solid frames) or **minimum dimension `< 80 px`** straight to `UNCERTAIN / NEEDS MANUAL REVIEW` with an explicit `uncertainty_reason` (verified across `1x1`, `60x60`, grayscale `L`, `RGBA`, and `CMYK` edge cases in `tests/test_end_to_end_qa.py`).

---

## Category 5: Video Deepfake Detection & Aggregation

### Q19. Why does `VideoForensicAnalyzer` explicitly report `temporal_modeling = False`?
**Answer:** Our module samples frames via OpenCV (`fps`, `interval`, or `uniform` modes) across MP4/AVI/MOV containers, extracts faces per frame, and batches them into `DualDomainSentinelNet`. Because it evaluates each frame independently without optical flow, 3D convolutions, or audio lip-sync verification, stating `temporal_modeling = False` prevents misleading examiners into assuming cross-frame motion modeling.

### Q20. How do you stop a single noisy or blurred video frame from flagging an entire authentic video as `MANIPULATED`?
**Answer:** `aggregate_video_scores()` uses two safeguards:
1. **10% Symmetric Trimmed Mean:** drops the highest and lowest $10\%$ of frame probabilities (`N >= 5`) to suppress isolated outliers;
2. **Sustained Multi-Frame Quorum:** classifies `MANIPULATED` only when at least $\max(2, \lceil 0.30\, N_{\text{face}} \rceil)$ frames exceed `uncertain_high` AND (`trimmed_mean >= tau` or `manip_ratio >= 0.45`). Single suspicious frames are flagged cleanly in the timeline (`isolated_spike_suppressed=True`) and route small clips to manual review rather than false manipulation calls. Zero-face videos never emit `AUTHENTIC` or `MANIPULATED`.

---

## Category 6: Explainable AI (XAI) & Defect Fixes

### Q21. Explain how your Grad-CAM handles a single-logit binary network cleanly (`B5` fix).
**Answer:** Prior `GradCAM.generate_cam()` ignored `target_class` on `(B, 1)` logits and kept `retain_graph=True`. We fixed it by backpropagating $+z$ when `target_class=1` (Manipulated evidence) and $-z$ when `target_class=0` (Authentic evidence), removing `retain_graph=True` to eliminate autograd memory leaks, pooling gradients globally across `spatial_stream.target_cam_layer` (`conv_head`), and enforcing spatial shape reconciliation inside `overlay_cam()`.

### Q22. Why must Error Level Analysis (ELA) never be presented as standalone proof of manipulation?
**Answer:** ELA computes $|I_{\text{orig}} - I_{\text{JPEG}(Q)}|$. Whole-frame diffusion/GAN images compressed uniformly show low spatial ELA mismatch (`0.1108` on our diffusion demo sample), WhatsApp/Instagram recompression erases prior splicing differentials (**washout**), and genuine crisp hair strands/text/eyeglass edges naturally yield bright residuals. Hence `app/xai_panel.py` shows `Q=95/90/75` comparisons strictly alongside `ELA_FORENSIC_LIMITATIONS`.

### Q23. What caused `FaceExtractor` to generate bogus face boxes on clothing text, and how did you fix it (`B4` fix)?
**Answer:** Previously, `if not has_eyes and skin_ratio < 0.20: return False` allowed a false Haar eye trigger (`minNeighbors=2`) on high-contrast t-shirt logos to pass even when `skin_ratio` was `~0%`. Raising eye `minNeighbors=3`, checking aspect ratio `[0.70, 1.40]` + area proportions, and enforcing an **unconditional `skin_ratio >= 0.14` floor** on non-grayscale color crops eliminated non-face clothing detections (`tests/test_multiface.py`).

### Q24. What was critical bug `B1` in `pipeline.py`?
**Answer:** `pipeline.py` called `spectral_details.get("peak_count", 0)`, whereas `SpectralAnalyzer.analyze_spectral_peaks()` returns `"detected_peaks"` (a list), never `"peak_count"`. Consequently `peak_count` evaluated to `0` everywhere and suppressed deconvolution spike alerts until fixed to `len(spectral_details.get("detected_peaks", []))`.

---

## Category 7: Forensic Integrity, Performance & Engineering QA

### Q25. Why do you hash raw uploaded file bytes separately from aligned `256x256` face crops?
**Answer:** Re-cropping and PNG-encoding a face crop discards EXIF metadata and changes byte layout, making it impossible to verify whether a candidate file matches the original uploaded evidence byte-for-byte (**B9**). Schema v2 stores both `raw_file_sha256` (streaming SHA-256 over raw upload bytes) and `image_sha256` (aligned crop PNG hash), enabling `verify_evidence_integrity()` to detect 1-byte tampering (`MISMATCH_TAMPERED`).

### Q26. How does SQLite enforce immutability on `audit_events`, and why distinguish software logs from legal chain of custody?
**Answer:** `src/core/database.py` creates `BEFORE UPDATE` and `BEFORE DELETE` SQLite triggers (`trg_audit_events_no_update`, `trg_audit_events_no_delete`) that execute `RAISE(ABORT, ...)`, turning `audit_events` (`upload`, `processing`, `prediction`, `report_generation`, `archive_action`) into an append-only SQL ledger. UI clearing performs soft archiving (`is_archived = 1`). Yet per `CHAIN_OF_CUSTODY_DISCLAIMER`, local SQLite logs still lack external **RFC 3161 TSA timestamps, PKI/HSM hardware signatures, and attested physical custody transfers** required by courts.

### Q27. How did you migrate legacy SQLite databases without losing existing records?
**Answer:** `ForensicDatabase.init_db()` inspects `PRAGMA table_info(forensic_examinations)` and runs idempotent `ALTER TABLE ... ADD COLUMN` statements for all 11 new Phase 8 columns, backfilling deterministic `EVD-LEGACY-...` IDs without touching any existing rows (`tests/test_database.py`).

### Q28. What speedups and fault-tolerance guarantees did Phase 10 add?
**Answer:**
1. Stacked mini-batch inference across multi-face crops and video frames (`safe_batched_forward`), reaching **`219.9 img/s`** on warmed Apple Silicon MPS (`batch_size=32`, `1.60x` vs `bs=1`, `26.4x` vs CPU `bs=32`) with `2.89e-06` parity against sequential `eval`.
2. Automatic `RuntimeError` recovery (`clear_accelerator_cache` + CPU fallback).
3. Instant Streamlit dashboard cache invalidation (`B10` fix) keyed on `(file_path, st_mtime_ns, st_size)` across 14 artifact files.

### Q29. What bug did Phase 11 catch during full Streamlit UI E2E testing, and how was it fixed?
**Answer:** Running `streamlit.testing.v1.AppTest` inside the full multi-module pytest session crashed on macOS with `RuntimeError: Cannot create a GUI FigureManager outside the main thread using the MacOS backend` because pyplot defaulted to interactive `MacOSX` before `Agg` was locked. Enforcing `matplotlib.use("Agg")` (`force=True` in `app/app.py`) across all plotting modules eliminated the worker-thread crash, enabling all **73/73** pytest checks (including live 6-section Streamlit E2E execution) to pass cleanly.

### Q30. What are your honest next research steps if given compute and external dataset access?
**Answer:**
1. Run `src/evaluation/external_benchmark.py` against downloaded **FaceForensics++ (c23/c40)**, **Celeb-DF**, and **DFDC** splits to measure true cross-generator AUC without leakage;
2. Unfreeze later EfficientNet stages with online spatial + spectral augmentations over real multi-identity corpora;
3. Extend `VideoForensicAnalyzer` from `temporal_modeling=False` to temporal landmark-consistency and audio-visual lip-sync modeling while retaining 3-outcome calibrated abstention.
