"""
Unit tests for VICReg projector and loss functions.
"""

import torch
import pytest
from v3.vicreg import VICRegProjector, VICRegLoss


def test_projector_shapes():
    proj = VICRegProjector(in_dim=256, hidden_dim=512, out_dim=512)
    z = torch.randn(8, 256)
    y = proj(z)
    assert y.shape == (8, 512)


def test_vicreg_loss_components():
    loss_fn = VICRegLoss(sim_weight=25.0, var_weight=25.0, cov_weight=1.0)
    y1 = torch.randn(16, 512, requires_grad=True)
    y2 = torch.randn(16, 512, requires_grad=True)

    loss, stats = loss_fn(y1, y2)
    assert loss.dim() == 0
    assert "sim_loss" in stats
    assert "var_loss" in stats
    assert "cov_loss" in stats
    assert "std_mean" in stats

    loss.backward()
    assert y1.grad is not None
    assert y2.grad is not None


def test_vicreg_identical_inputs():
    loss_fn = VICRegLoss()
    y = torch.randn(32, 512)
    loss, stats = loss_fn(y, y)
    # Invariance loss should be 0 for identical inputs
    assert stats["sim_loss"] < 1e-6
