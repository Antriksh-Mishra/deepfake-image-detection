"""
Unit tests for Spatial, Frequency, and Cross-Attention Fusion Neural Networks.
"""

import pytest
import torch
from src.models.spatial_backbone import SpatialBackbone
from src.models.frequency_stream import FrequencyStream, get_srm_filters
from src.models.dual_domain_net import DualDomainSentinelNet, CrossAttentionFusion
from src.models.loss import BinaryFocalLoss, LabelSmoothingBCELoss


def test_srm_filters_shape():
    filters = get_srm_filters()
    assert filters.shape == (3, 1, 5, 5)


def test_spatial_backbone():
    backbone = SpatialBackbone(backbone_name="efficientnet_b0", pretrained=False, feat_dim=512)
    x = torch.randn(2, 3, 256, 256)
    out = backbone(x)
    assert out.shape == (2, 512)


def test_frequency_stream():
    freq_net = FrequencyStream(in_channels=1, feat_dim=256)
    x = torch.randn(2, 1, 256, 256)
    out = freq_net(x)
    assert out.shape == (2, 256)


def test_cross_attention_fusion():
    fusion = CrossAttentionFusion(spatial_dim=512, spectral_dim=256, embed_dim=256, num_heads=4)
    s = torch.randn(2, 512)
    f = torch.randn(2, 256)
    fused, weights = fusion(s, f)
    assert fused.shape == (2, 256)
    assert weights is not None


def test_dual_domain_net_forward():
    model = DualDomainSentinelNet(backbone_name="efficientnet_b0", pretrained=False, fusion_type="cross_attention")
    s = torch.randn(2, 3, 256, 256)
    f = torch.randn(2, 1, 256, 256)
    logits = model(s, f)
    assert logits.shape == (2, 1)


def test_loss_functions():
    focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    smooth = LabelSmoothingBCELoss(smoothing=0.05)
    
    logits = torch.randn(4, 1, requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    
    loss_f = focal(logits, targets)
    loss_s = smooth(logits, targets)
    
    assert loss_f.item() >= 0.0
    assert loss_s.item() >= 0.0
    
    # Check backward flow
    loss_f.backward()
    assert logits.grad is not None
