# Forensic & Mathematical Methodology — Deepfake Forensic Sentinel

**Author:** Antriksh Mishra (Final Year B.Tech CSE Capstone)  
**Scope:** Exact mathematical, statistical, and algorithmic formulation corresponding directly to `src/` and `app/`.

---

## 1. Dual-Domain Representation Learning

### 1.1 Spatial Domain Stream (`SpatialBackbone`)
Given an aligned facial crop $I \in \mathbb{R}^{256 \times 256 \times 3}$, pixels are normalized using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) and passed through a convolutional `efficientnet_b0` trunk (`4,007,548` parameters):

$$\mathbf{h}_s = \text{Flatten}(\text{AdaptiveAvgPool}_{1 \times 1}(f_{\text{eff}}(I))) \in \mathbb{R}^{1280}$$

$$\mathbf{f}_s = \text{ReLU}(\text{LayerNorm}(W_s \mathbf{h}_s + \mathbf{b}_s)) \in \mathbb{R}^{512}$$

During production training ([`src/train.py`](../src/train.py)), $f_{\text{eff}}$ is frozen and precomputed once (`spatial_flat`), while $W_s, \mathbf{b}_s$ (`656,896` parameters) remain trainable.

---

### 1.2 Frequency Domain & SRM Steganalysis Stream (`SpectralAnalyzer` + `FrequencyStream`)

Generator upsampling layers (transposed convolutions, nearest-neighbor interpolation followed by convolution, or pixel shuffle) introduce periodic spatial modulation when kernel width $K$ and stride $S$ produce uneven pixel overlap. In luminance space $I_{\text{gray}} \in \mathbb{R}^{256 \times 256}$, the centered 2D Discrete Fourier Transform is computed in [`src/explainability/spectral_analysis.py`](../src/explainability/spectral_analysis.py):

$$F(u, v) = \sum_{x=0}^{255} \sum_{y=0}^{255} I_{\text{gray}}(x, y)\, e^{-j 2\pi \left(\frac{u x}{256} + \frac{v y}{256}\right)}$$

$$S_{\log}(u, v) = \ln\left(1 + \left|F_{\text{shift}}(u, v)\right|\right)$$

$S_{\log}$ is min-max scaled to $[0, 255]$ (`uint8`) and normalized to $[-1, 1]$ via `SPECTRAL_TRANSFORM` ($\mu=0.5, \sigma=0.5$), yielding tensor $X_f \in \mathbb{R}^{1 \times 256 \times 256}$.

To amplify high-frequency spectral discontinuities while suppressing smooth low-frequency decay, [`src/models/frequency_stream.py`](../src/models/frequency_stream.py) convolves $X_f$ with **three fixed $5 \times 5$ Spatial Rich Model (SRM) high-pass kernels** stored as non-trainable buffers:

$$W_1 = \frac{1}{4}\begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & -1 & 2 & -1 & 0 \\ 0 & 2 & -4 & 2 & 0 \\ 0 & -1 & 2 & -1 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}, \quad
W_2 = \frac{1}{12}\begin{bmatrix} -1 & 2 & -2 & 2 & -1 \\ 2 & -6 & 8 & -6 & 2 \\ -2 & 8 & -12 & 8 & -2 \\ 2 & -6 & 8 & -6 & 2 \\ -1 & 2 & -2 & 2 & -1 \end{bmatrix}, \quad
W_3 = \frac{1}{4}\begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & -1 & 0 & 0 \\ 0 & -1 & 4 & -1 & 0 \\ 0 & 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$

Concatenating $X_f$ and the 3 SRM response maps yields a 4-channel spectral tensor:

$$\tilde{X}_f = [X_f;\, X_f * W_1;\, X_f * W_2;\, X_f * W_3] \in \mathbb{R}^{4 \times 256 \times 256}$$

which passes through a 4-stage stride-2 `Conv2d-BatchNorm-LeakyReLU(0.2)` encoder ($4 \to 32 \to 64 \to 128 \to 256$), global average pooling, and linear LayerNorm projection to produce $\mathbf{f}_f \in \mathbb{R}^{256}$.

---

### 1.3 Multi-Head Cross-Attention Fusion & MLP Head (`CrossAttentionFusion`)

Defined in [`src/models/dual_domain_net.py`](../src/models/dual_domain_net.py), both streams are projected into a shared $d_e = 256$ token space (`unsqueeze(1)` length-1 sequence):

$$\mathbf{s}_e = W_{p,s}\mathbf{f}_s + \mathbf{b}_{p,s} \in \mathbb{R}^{1 \times 256}, \qquad \mathbf{f}_e = W_{p,f}\mathbf{f}_f + \mathbf{b}_{p,f} \in \mathbb{R}^{1 \times 256}$$

Spatial queries attend over spectral keys and values across $H = 4$ heads ($d_k = 64$):

$$\mathbf{a} = \text{MultiHeadAttn}(Q=\mathbf{s}_e,\, K=\mathbf{f}_e,\, V=\mathbf{f}_e)$$

$$\mathbf{x} = \text{LayerNorm}(\mathbf{s}_e + \mathbf{a}), \qquad \mathbf{z}_{\text{fused}} = \text{LayerNorm}(\mathbf{x} + \text{MLP}_{\text{GELU}}(\mathbf{x})) \in \mathbb{R}^{256}$$

Finally, the 3-layer classification head ($256 \to 128 \to 32 \to 1$) emits a scalar raw logit $z \in \mathbb{R}$.

---

## 2. Training Objectives (`src/models/loss.py`)

Both loss modules are implemented cleanly:

1. **`LabelSmoothingBCELoss(smoothing=0.06)` (Active in [`src/train.py`](../src/train.py) & Ablation Exp 5):**
   Transforms hard binary targets $y \in \{0, 1\}$ to smoothed targets $\tilde{y} = y(1 - 2\epsilon) + \epsilon$ ($\epsilon = 0.06$, i.e., $\{0.06, 0.94\}$) and minimizes:

   $$\mathcal{L}_{\text{LS-BCE}}(z, y) = -\left[\tilde{y}\ln\sigma(z) + (1 - \tilde{y})\ln(1 - \sigma(z))\right]$$

2. **`BinaryFocalLoss(alpha=0.50, gamma=2.0)` (Available alternative):**

   $$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t), \quad p_t = y\sigma(z) + (1 - y)(1 - \sigma(z))$$

Optimization in `src/train.py` uses `AdamW` ($\text{lr}=3\times 10^{-4}$, $\text{weight\_decay}=10^{-3}$), gradient norm clipping (`max_norm=1.0`), `CosineAnnealingLR` ($\eta_{\min}=10^{-6}$), and validation ROC-AUC checkpoint selection.

---

## 3. Decoupled Classifier vs. Auxiliary Forensic Cues

Prior to Phase 5 (**B2 defect**), legacy code linearly overrode neural probability using unvalidated weights (`0.65N + 0.18S + 0.17E`). That override was permanently eliminated:

- **Primary Verdict Probability** comes exclusively from temperature-scaled `DualDomainSentinelNet` logits.
- **Auxiliary Spectral Score (`p_spectral`) and ELA Score (`p_ela`)** in `ForensicInferencePipeline._evaluate_auxiliary_forensics()` are strictly explanatory telemetry fields (`evidence_decoupled_from_classifier = True`, `heuristic_override_applied = False`).

---

## 4. Validation-Only Calibration & 3-Outcome Abstention (`src/evaluation/calibration.py`)

All calibration fitting operates exclusively on validation logits $\{z_i, y_i\}_{i=1}^{N_{\text{val}}}$ (`data/val`, $N=400$), guaranteeing `zero_test_set_tuning = True` and leaving `checkpoints/best_model.pth` unmodified.

### 4.1 Scalar Temperature Scaling
Minimizes validation Negative Log-Likelihood (NLL) over $T \in [0.20, 15.0]$ via dense grid initialization followed by CPU `float32` `L-BFGS`:

$$T^* = \arg\min_{T > 0} \left[-\frac{1}{N_{\text{val}}}\sum_{i=1}^{N_{\text{val}}} \Big(y_i \ln \sigma(z_i / T) + (1 - y_i)\ln(1 - \sigma(z_i / T))\Big)\right]$$

Yields validated **$T^* = 2.654271$**, reducing validation NLL from `0.772787` to `0.609668`.

### 4.2 Calibration Quality Metrics
- **Brier Score:** $\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2$.
- **Positive-Class Expected Calibration Error ($\text{ECE}_{\text{pos}}$):** Partitions $\hat{p}_i \in [0, 1]$ into $M=10$ bins $B_m$ and computes $\sum_{m=1}^{M}\frac{|B_m|}{N}\left|\bar{y}(B_m) - \bar{p}(B_m)\right|$.
- **Top-Label Confidence ECE ($\text{ECE}_{\text{conf}}$):** Evaluates Guo et al. confidence $\hat{c}_i = \max(\hat{p}_i, 1 - \hat{p}_i)$ against empirical binary accuracy over $M=10$ bins on $[0.5, 1.0]$.

### 4.3 Operating Threshold & 3-Outcome Abstention Policy
1. **Validation Operating Threshold ($\tau^*$):** Sweeps $[0.15, 0.85]$ on calibrated validation probabilities to maximize Youden's $J$ statistic:

   $$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 = \text{TPR}(\tau) - \text{FPR}(\tau)$$

   Selecting **$\tau^* = 0.71$** (matching validation best-F1 threshold `0.71`, lifting validation accuracy from `69.00%` to `77.25%`).

2. **Uncertainty Band Search (`fit_uncertainty_policy_on_validation`):** Searches asymmetric quantile & logit-margin neighborhoods `[uncertain_low, uncertain_high]` around $\tau^*$ to maximize `selective_accuracy + 0.12 * coverage` subject to non-degenerate class support and `coverage >= 0.70`, obtaining **$[\tau_{\text{low}}, \tau_{\text{high}}] = [0.7070, 0.7188]$**.

3. **Strict Decision Invariant (Eliminating False-Negative Flaws):**
   Prior to the B11 fix, evaluating $P < \tau_{\text{low}}$ as authentic caused elevated predictions ($P = 70.25\% > 50\%$) to be declared `AUTHENTIC`. The revised engine enforces a strict mathematical invariant:
   - **`REAL` / `AUTHENTIC`**: Requires $\hat{p}_{\text{cal}} < \tau_{\text{low}}$ **AND** $\hat{p}_{\text{cal}} \le 0.50$ (i.e. $P(\text{Authentic}) \ge 50\%$).
   - **`FAKE` / `MANIPULATED`**: Requires $\hat{p}_{\text{cal}} > \tau_{\text{high}}$.
   - **`UNCERTAIN`**: Triggered when $\tau_{\text{low}} \le \hat{p}_{\text{cal}} \le \tau_{\text{high}}$, **OR** when $\hat{p}_{\text{cal}} > 0.50$ but below $\tau_{\text{high}}$, **OR** when image-quality guards fail. Under this invariant, an image with $P(\text{Manipulated}) > 50\%$ can never be ruled `REAL`.

4. **Image-Quality Abstention Guards:** Computes Laplacian variance $\text{Var}(\nabla^2 I_{\text{gray}})$ and input dimensions $(W, H)$. Predictions automatically route to `UNCERTAIN` if:
   - $\text{Var}(\nabla^2 I_{\text{gray}}) < 15.0$, OR
   - $\min(W, H) < 80\text{ px}$.

5. **Confidence & Predictive Entropy:**

   $$c_{\text{cls}} = \max(\hat{p}_{\text{cal}}, 1 - \hat{p}_{\text{cal}}), \qquad H(\hat{p}_{\text{cal}}) = -\hat{p}_{\text{cal}}\log_2 \hat{p}_{\text{cal}} - (1 - \hat{p}_{\text{cal}})\log_2(1 - \hat{p}_{\text{cal}})$$

---

## 5. Explainable AI Formulations (`src/explainability/`)

### 5.1 Spatial Grad-CAM (`gradcam.py`)
Hooks the final spatial convolutional block (`features[-1]` = `conv_head`) with forward activations $A^k \in \mathbb{R}^{H' \times W'}$ and backward gradients (cleanly freeing graphs, zero `retain_graph=True` leak — **B5 fix**). For single-logit output $z$:
- `target_class = 1` (Manipulated): differentiates $z$;
- `target_class = 0` (Authentic): differentiates $-z$.

$$w_k = \frac{1}{H' W'}\sum_{i,j} \frac{\partial s}{\partial A^k_{i,j}}, \qquad L_{\text{CAM}} = \text{ReLU}\!\left(\sum_k w_k A^k\right)$$

Min-max normalized and bicubic upsampled to `256x256`. 
- **Facial Images (`has_face=True`):** Regional pooling summarizes activation means across upper ocular ($0\text{--}45\%$ height), mid-facial / nasolabial ($45\text{--}75\%$), and jawline / perimeter ($75\text{--}100\%$).
- **Non-Face Objects (`has_face=False`):** Regional pooling transitions to structural object zones: Upper Structural Perimeter ($0\text{--}35\%$), Central Object Core / Foreground ($35\text{--}70\%$), and Lower Base / Support Boundary ($70\text{--}100\%$).

### 5.2 Spatial SRM Noise Residual Analysis (`srm_analysis.py`)
Computes independent 2D high-pass filtering directly on input luminance $I$ using three $5 \times 5$ SRM kernels with reflection padding (`cv2.BORDER_REFLECT_101`):
- Residual Energy: $E_k = \frac{1}{HW}\sum_{x,y} R_k(x,y)^2$
- Residual Kurtosis: $\kappa = \frac{\mathbb{E}[(R - \mu)^4]}{\sigma^4}$ (Gaussian noise baseline $\approx 3.0$)
- Spatial Noise Inconsistency: Coefficient of variation ($\text{CV} = \sigma / \mu$) across non-overlapping $16 \times 16$ spatial blocks.

### 5.3 Azimuthal Radial Spectral Profile & Optical Decay Baseline (`spectral_analysis.py`)
Integrates $S_{\log}(u, v)$ radially at integer pixel radius $r = \lfloor \sqrt{(u - u_0)^2 + (v - v_0)^2} \rfloor$:

$$P(r) = \frac{1}{|R_r|}\sum_{(u,v)\in R_r} S_{\log}(u, v)$$

Dynamically fits an adaptive optical decay power-law baseline $\hat{P}_{\text{optical}}(r) = A - \alpha \ln(1 + r)$ to the image's own low-frequency bands ($r \in [2, 35]$), preventing false frequency alerts on non-face objects with sharp organic boundaries.

### 5.4 Container-Aware Error Level Analysis (`ela.py`)
Re-encodes the RGB crop at JPEG quality $Q \in \{90\}$:

$$E(x, y) = \left|I_{\text{orig}}(x, y) - I_{\text{JPEG}(Q)}(x, y)\right|$$

Automatically detects source container format (`PNG`, `WEBP`, `JPEG`). For lossless formats (such as PNG), compression error represents lossy recompression approximation rather than original compression history, qualifying all derived quadrant scores accordingly.

---

## 6. Dataset Integrity & Video Aggregation Formulas

### 6.1 Perceptual Hashing & Leakage Control (`src/data/dataset_audit.py`)
- Exact deduplication: raw file SHA-256.
- Perceptual near-duplicate detection: `32x32` grayscale Lanczos resize $\to$ orthonormal 2D DCT (`scipy.fft.dct`) $\to$ top-left `8x8` coefficients thresholded against AC median $\to$ 64-bit binary digest compared via Hamming distance ($\le 4$).
- Disjoint partitioning: `sklearn.model_selection.StratifiedGroupKFold` ensures zero `source_group` overlap across splits.

### 6.2 Frame-Level Video Aggregation (`src/video/video_analyzer.py`)
Across $N_{\text{face}}$ evaluated face-containing frames (`probs` sorted as $p_{(1)} \le \dots \le p_{(N_{\text{face}})}$):
- **10% Symmetric Trimmed Mean:** drops $k = \lfloor 0.10 N_{\text{face}} \rfloor$ extreme frames on each tail ($N_{\text{face}} \ge 5$; median if $N_{\text{face}} < 5$):

  $$\bar{p}_{\text{trim}} = \frac{1}{N_{\text{face}} - 2k}\sum_{i=k+1}^{N_{\text{face}}-k} p_{(i)}$$

- **Sustained Multi-Frame Quorum:** requires $N_{\text{manip}} \ge \max\big(2,\; \lceil 0.30\, N_{\text{face}} \rceil\big)$ AND $(\bar{p}_{\text{trim}} \ge \tau \text{ or } N_{\text{manip}}/N_{\text{face}} \ge 0.45)$ to return a video-level `MANIPULATED` verdict, mathematically preventing any isolated single-frame glitch from triggering a video false positive.
