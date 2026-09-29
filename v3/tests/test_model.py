"""
Unit tests for CNN encoder, Local branch, Global branch, VirtualPatchModule, and JigsawSolverV3.
"""

import torch
import pytest
from v3.config import ModelConfig
from v3.cnn_encoder import PatchCNNEncoder
from v3.model import JigsawSolverV3, VirtualPatchModule, LocalBranch, GlobalBranch


def test_cnn_encoder_shapes():
    encoder = PatchCNNEncoder(content_dim=256, stages=(64, 128, 256))
    x_5d = torch.randn(2, 25, 3, 13, 13)
    out_5d = encoder(x_5d)
    assert out_5d.shape == (2, 25, 256)

    x_4d = torch.randn(10, 3, 13, 13)
    out_4d = encoder(x_4d)
    assert out_4d.shape == (10, 256)


def test_virtual_patch_module():
    vpm = VirtualPatchModule(content_dim=256, temperature=0.1)
    H_cand = torch.randn(4, 25, 256)
    z_virt, weights = vpm(H_cand)

    assert z_virt.shape == (4, 256)
    assert weights.shape == (4, 25)
    # Weights should sum to 1.0 along candidate dimension
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_solver_v3_forward_and_backward():
    cfg = ModelConfig(
        content_dim=256,
        local_layers=2,
        local_heads=8,
        global_layers=4,
        global_heads=8,
        global_ffn_dim=1024,
        gradient_checkpointing=False
    )
    model = JigsawSolverV3(cfg)
    model.train()

    B = 2
    K = 25
    patches = torch.randn(B, K, 3, 13, 13)
    raw_feats = model.encode_candidates(patches)
    assert raw_feats.shape == (B, K, 256)

    grid_placed = torch.full((B, 5, 5), -1, dtype=torch.long)
    grid_placed[:, 2, 2] = 12
    used_cand = torch.zeros((B, K), dtype=torch.bool)
    used_cand[:, 12] = True
    cand_to_slot = torch.full((B, K, 2), -1, dtype=torch.long)
    cand_to_slot[:, 12] = torch.tensor([2, 2])
    target_coords = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)

    scores, H_cand = model.score_step(
        raw_feats=raw_feats,
        grid_placed=grid_placed,
        used_candidates=used_cand,
        cand_to_slot=cand_to_slot,
        target_coords=target_coords,
        grid_size=5
    )
    assert scores.shape == (B, K)
    assert H_cand.shape == (B, K, 256)

    # Virtual patch
    cand_to_slot_full = torch.zeros((B, K, 2), dtype=torch.long)
    used_cand_full = torch.ones((B, K), dtype=torch.bool)
    z_virt, attn = model.compute_virtual_patch(raw_feats, cand_to_slot_full, used_cand_full)

    loss = scores.sum() + z_virt.sum()
    loss.backward()

    # Verify gradients flow into CNN encoder and virtual query token
    assert model.encoder.conv0.weight.grad is not None
    assert model.virtual_patch.q_virtual.grad is not None


def test_global_transformer_gradient_checkpointing_path():
    branch = GlobalBranch(
        content_dim=32,
        num_heads=4,
        num_layers=2,
        ffn_dim=64,
        gradient_checkpointing=True
    )
    branch.train()
    x = torch.randn(2, 25, 32, requires_grad=True)
    output = branch.run_transformer(x)
    output.square().mean().backward()

    assert x.grad is not None
    assert branch.transformer.layers[0].linear1.weight.grad is not None
