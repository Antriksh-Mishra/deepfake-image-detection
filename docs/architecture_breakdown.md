# Mathematical & Architectural Quick Reference

> **Note:** For exhaustive Phase 12 engineering specifications, calibrated equations, and empirical tables, see [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) and [`METHODOLOGY.md`](METHODOLOGY.md).

---

## 1. The Generator Artifact Problem: Why Frequency Domain?

Standard Convolutional Neural Networks (CNNs) trained purely on RGB pixel values can latch onto donor color balance or lighting quirks (`59.05%` accuracy, `77.37%` FPR in our Spatial-Only ablation). Generative upsampling operations (transposed convolutions and sub-pixel convolutions) leave periodic spatial modulation whenever filter kernels overlap unevenly across strides. Applying the centered **2D Discrete Fourier Transform** concentrates periodic grids into spectral peaks:

$$S_{\log}(u, v) = \ln\!\left(1 + \left|\sum_{x=0}^{255}\sum_{y=0}^{255} I_{\text{gray}}(x, y)\, e^{-j 2\pi \left(\frac{ux}{256} + \frac{vy}{256}\right)}\right|\right)$$

In `src/explainability/spectral_analysis.py`, the 1D azimuthal radial mean $P(r)$ is benchmarked against a calibrated log-decay baseline $\hat{P}(r) = \max(0, P(1) + 1.69\ln 2 - 1.69\ln(1+r))$ and local reference portrait curves, while `src/models/frequency_stream.py` applies **3 fixed $5\times 5$ SRM high-pass filters** ($W_1/4$, $W_2/12$, $W_3/4$) before a 4-stage stride-2 convolutional encoder (`256`-d).

---

## 2. Multi-Head Cross-Attention Fusion

Rather than static vector concatenation, `CrossAttentionFusion` (`src/models/dual_domain_net.py`) projects 512-d spatial features ($\mathbf{f}_s$) and 256-d spectral features ($\mathbf{f}_f$) into $d_e = 256$ tokens across $4$ attention heads:

$$\mathbf{a} = \text{MultiHeadAttn}(Q=\mathbf{s}_e,\; K=\mathbf{f}_e,\; V=\mathbf{f}_e), \qquad \mathbf{x} = \text{LayerNorm}(\mathbf{s}_e + \mathbf{a}), \qquad \mathbf{z}_{\text{fused}} = \text{LayerNorm}(\mathbf{x} + \text{MLP}_{\text{GELU}}(\mathbf{x}))$$

---

## 3. Training Objectives & Validation Temperature Scaling

- **Production Training Loss (`src/train.py`):** `LabelSmoothingBCELoss(smoothing=0.06)` maps hard targets $\{0, 1\} \to [0.06, 0.94]$ to mitigate overconfidence (`BinaryFocalLoss(alpha=0.50, gamma=2.0)` is also implemented cleanly in `src/models/loss.py`).
- **Validation-Only Calibration (`src/evaluation/calibration.py`):** Temperature scaling fits $T^* = 2.654271$ strictly on validation logits ($\hat{p}_{\text{cal}} = \sigma(z / T^*)$) with Youden's J threshold $\tau^* = 0.71$ and 3-outcome band $[0.7070, 0.7188]$. Auxiliary spectral/ELA indicators never override classifier logits (`heuristic_override_applied = False`).

---

## 4. Error Level Analysis (ELA)

`src/explainability/ela.py` re-compresses face crops at JPEG quality $Q = 90$ (plus multi-quality comparisons at $Q \in \{95, 90, 75\}$):

$$E(x, y) = \left|I_{\text{orig}}(x, y) - I_{\text{JPEG}(Q)}(x, y)\right|$$

Evaluating quadrant mean discrepancy ratio and $16\times 16$ block CV, always paired with explicit non-proof caveats regarding messaging-app recompression washout and natural high-frequency edge responses.
