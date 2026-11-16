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

from typing import Optional, Tuple
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
    ) -> torch.Tensor:
        """
        Forward pass supporting raw tensors or pre-pooled `spatial_flat` features.

        Returns:
            logits: (B, 1) raw logit output.
        """
        if self.fusion_type == "frequency_only":
            if spectral_tensor is None:
                raise ValueError("spectral_tensor is required when fusion_type='frequency_only'.")
            assert self.frequency_stream is not None
            fused = self.frequency_stream(spectral_tensor)
            return self.classifier(fused)

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
            return self.classifier(s_feat)

        if spectral_tensor is None:
            spectral_tensor = torch.zeros((b, 1, 256, 256), device=device)

        assert self.frequency_stream is not None
        f_feat = self.frequency_stream(spectral_tensor)

        if self.fusion_type == "cross_attention":
            assert self.fusion is not None
            fused, _ = self.fusion(s_feat, f_feat)
        else:  # concat
            fused = torch.cat([s_feat, f_feat], dim=1)

        return self.classifier(fused)

    def predict_proba(self, spatial_tensor: torch.Tensor, spectral_tensor: torch.Tensor) -> float:
        """Helper to get sigmoid probability [0.0, 1.0] of image being a deepfake."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(spatial_tensor, spectral_tensor)
            proba = torch.sigmoid(logits).item()
        return proba
