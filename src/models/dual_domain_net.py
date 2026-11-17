"""
Dual-Domain Cross-Attention Neural Network for Deepfake Detection.
Fuses fine-grained spatial features (EfficientNet/ResNet) with forensic
spectral representations (2D FFT + SRM) via Multi-Head Cross-Attention.

Supports four fusion/ablation configurations via `fusion_type`:
- "cross_attention": Dual-domain spatial + spectral fused via Multi-Head Cross-Attention (default/production)
- "concat": Dual-domain spatial + spectral simple feature concatenation
- "spatial_only": Spatial RGB stream only (ablation baseline)
- "frequency_only": Frequency SRM + 2D FFT stream only (ablation baseline)
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.spatial_backbone import SpatialBackbone
from src.models.frequency_stream import FrequencyStream


class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion layer.
    Allows spatial tokens to query spectral frequency tokens,
    dynamically weighting domain confidence.
    """
    def __init__(self, spatial_dim: int = 512, spectral_dim: int = 256, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.proj_spatial = nn.Linear(spatial_dim, embed_dim)
        self.proj_spectral = nn.Linear(spectral_dim, embed_dim)

        # Spatial queries attend to Spectral keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, spatial_feat: torch.Tensor, spectral_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_emb = self.proj_spatial(spatial_feat).unsqueeze(1)
        f_emb = self.proj_spectral(spectral_feat).unsqueeze(1)

        attn_out, attn_weights = self.cross_attn(query=s_emb, key=f_emb, value=f_emb)
        x = self.norm1(s_emb + attn_out)
        x = self.norm2(x + self.mlp(x))
        fused = x.squeeze(1)
        return fused, attn_weights


class DualDomainSentinelNet(nn.Module):
    """
    Complete Dual-Domain Deepfake Detection Architecture with ablation mode support.
    """
    SUPPORTED_FUSIONS = ("cross_attention", "concat", "spatial_only", "frequency_only")

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        spatial_dim: int = 512,
        spectral_dim: int = 256,
        embed_dim: int = 256,
        num_heads: int = 4,
        fusion_type: str = "cross_attention",
        dropout: float = 0.3,
    ):
        super().__init__()
        if fusion_type not in self.SUPPORTED_FUSIONS:
            raise ValueError(
                f"Unsupported fusion_type: '{fusion_type}'. Expected one of {self.SUPPORTED_FUSIONS}."
            )
        self.fusion_type = fusion_type
        self.backbone_name = backbone_name

        # 1. Spatial Stream (present for cross_attention, concat, spatial_only)
        if fusion_type in ("cross_attention", "concat", "spatial_only"):
            self.spatial_stream: Optional[SpatialBackbone] = SpatialBackbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                feat_dim=spatial_dim,
            )
        else:
            self.spatial_stream = None

        # 2. Frequency Stream (present for cross_attention, concat, frequency_only)
        if fusion_type in ("cross_attention", "concat", "frequency_only"):
            self.frequency_stream: Optional[FrequencyStream] = FrequencyStream(
                in_channels=1,
                feat_dim=spectral_dim,
            )
        else:
            self.frequency_stream = None

        # 3. Fusion Layer
        if fusion_type == "cross_attention":
            self.fusion: Optional[CrossAttentionFusion] = CrossAttentionFusion(
                spatial_dim=spatial_dim,
                spectral_dim=spectral_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            classifier_in = embed_dim
        elif fusion_type == "concat":
            self.fusion = None
            classifier_in = spatial_dim + spectral_dim
        elif fusion_type == "spatial_only":
            self.fusion = None
            classifier_in = spatial_dim
        else:  # frequency_only
            self.fusion = None
            classifier_in = spectral_dim

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        spatial_tensor: Optional[torch.Tensor] = None,
        spectral_tensor: Optional[torch.Tensor] = None,
        spatial_flat: Optional[torch.Tensor] = None,
        return_branch_details: bool = False,
    ):
        """
        Forward pass supporting raw tensors or pre-pooled `spatial_flat` features.

        Args:
            spatial_tensor: (B, 3, H, W) normalized RGB tensor.
            spectral_tensor: (B, 1, H, W) normalized 2D DFT log-magnitude spectrum.
            spatial_flat: (B, 1280) pre-pooled features from frozen EfficientNet trunk.
            return_branch_details: if True, returns (logits, branch_dict) with decoupled branch signals.

        Returns:
            logits: (B, 1) raw logit output (or tuple of logits and branch_dict if return_branch_details=True).
        """
        if self.fusion_type == "frequency_only":
            if spectral_tensor is None:
                raise ValueError("spectral_tensor is required when fusion_type='frequency_only'.")
            assert self.frequency_stream is not None
            fused = self.frequency_stream(spectral_tensor)
            logits = self.classifier(fused)
            if return_branch_details:
                return logits, {
                    "spatial_logit": None,
                    "frequency_logit": logits,
                    "fused_logit": logits,
                    "attn_weights": None,
                }
            return logits

        assert self.spatial_stream is not None
        if spatial_flat is not None:
            s_feat = self.spatial_stream.proj(spatial_flat)
            device = spatial_flat.device
            b = spatial_flat.shape[0]
        else:
            if spatial_tensor is None:
                raise ValueError("Either spatial_tensor or spatial_flat must be provided.")
            s_feat = self.spatial_stream(spatial_tensor)
            device = spatial_tensor.device
            b = spatial_tensor.shape[0]

        if self.fusion_type == "spatial_only":
            logits = self.classifier(s_feat)
            if return_branch_details:
                return logits, {
                    "spatial_logit": logits,
                    "frequency_logit": None,
                    "fused_logit": logits,
                    "attn_weights": None,
                }
            return logits

        if spectral_tensor is None:
            spectral_tensor = torch.zeros((b, 1, 256, 256), device=device)

        assert self.frequency_stream is not None
        f_feat = self.frequency_stream(spectral_tensor)

        attn_weights = None
        if self.fusion_type == "cross_attention":
            assert self.fusion is not None
            fused, attn_weights = self.fusion(s_feat, f_feat)
        else:  # concat
            fused = torch.cat([s_feat, f_feat], dim=1)

        logits = self.classifier(fused)

        if return_branch_details:
            branch_dict = self._compute_branch_logits(s_feat, f_feat, logits, attn_weights)
            return logits, branch_dict

        return logits

    def _compute_branch_logits(
        self,
        s_feat: torch.Tensor,
        f_feat: torch.Tensor,
        fused_logits: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Calculates decoupled spatial and frequency branch contributions via branch zero-ablation."""
        if self.fusion_type == "cross_attention":
            assert self.fusion is not None
            zero_f = torch.zeros_like(f_feat)
            zero_s = torch.zeros_like(s_feat)
            fused_s_only, _ = self.fusion(s_feat, zero_f)
            spatial_logit = self.classifier(fused_s_only)

            fused_f_only, _ = self.fusion(zero_s, f_feat)
            frequency_logit = self.classifier(fused_f_only)
        elif self.fusion_type == "concat":
            zero_f = torch.zeros_like(f_feat)
            zero_s = torch.zeros_like(s_feat)
            spatial_logit = self.classifier(torch.cat([s_feat, zero_f], dim=1))
            frequency_logit = self.classifier(torch.cat([zero_s, f_feat], dim=1))
        else:
            spatial_logit = fused_logits
            frequency_logit = fused_logits

        return {
            "spatial_logit": spatial_logit,
            "frequency_logit": frequency_logit,
            "fused_logit": fused_logits,
            "attn_weights": attn_weights,
        }

    def compute_branch_predictions(
        self,
        spatial_tensor: Optional[torch.Tensor] = None,
        spectral_tensor: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Convenience method returning calibrated branch probabilities and discrepancy flags."""
        self.eval()
        with torch.no_grad():
            fused_logits, b_dict = self.forward(
                spatial_tensor=spatial_tensor,
                spectral_tensor=spectral_tensor,
                return_branch_details=True,
            )
            t = max(float(temperature), 1e-6)
            fused_z = float(fused_logits.reshape(-1)[0].item())
            fused_p = float(torch.sigmoid(torch.tensor(fused_z / t)).item())

            s_z = float(b_dict["spatial_logit"].reshape(-1)[0].item()) if b_dict["spatial_logit"] is not None else fused_z
            s_p = float(torch.sigmoid(torch.tensor(s_z / t)).item())

            f_z = float(b_dict["frequency_logit"].reshape(-1)[0].item()) if b_dict["frequency_logit"] is not None else fused_z
            f_p = float(torch.sigmoid(torch.tensor(f_z / t)).item())

            discrepancy = abs(s_p - f_p)
            has_conflict = (s_p >= 0.50 and f_p < 0.50) or (f_p >= 0.50 and s_p < 0.50)
            false_negative_risk = bool(f_p >= 0.50 and fused_p < 0.50)

            return {
                "fused_logit": fused_z,
                "fused_prob": fused_p,
                "spatial_logit": s_z,
                "spatial_prob": s_p,
                "frequency_logit": f_z,
                "frequency_prob": f_p,
                "discrepancy": discrepancy,
                "has_conflict": has_conflict,
                "false_negative_risk": false_negative_risk,
                "consensus_prob": float(0.40 * fused_p + 0.35 * f_p + 0.25 * s_p),
                "attn_weights": b_dict.get("attn_weights"),
            }

    def predict_proba(self, spatial_tensor: torch.Tensor, spectral_tensor: torch.Tensor) -> float:
        """Helper to get sigmoid probability [0.0, 1.0] of image being a deepfake."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(spatial_tensor, spectral_tensor)
            proba = torch.sigmoid(logits).item()
        return proba
