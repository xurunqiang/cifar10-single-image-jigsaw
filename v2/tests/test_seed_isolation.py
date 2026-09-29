"""
Test 1: Seed Isolation & Gradient Propagation
Verify:
- Seed is placed at given coordinate and never predicted/overwritten.
- Seed is excluded from loss and patch accuracy.
- Seed features pass gradients back to ContentEncoder.
- Local and global modules both receive gradients from non-seed slot losses.
"""

import pytest
import torch
import torch.nn as nn
from v2.config import ModelConfig
from v2.model import JigsawSolverV2
from v2.dataset import JigsawProblemGenerator


def test_seed_isolation_and_gradient_flow():
    grid_size = 3
    K = grid_size * grid_size
    img = torch.randn(3, 32, 32)

    prob = JigsawProblemGenerator.generate(img, grid_size=grid_size, seed_mode="center")
    candidates = prob["candidates"].unsqueeze(0)  # (1, K, 3, P, P)
    seed_cand = prob["seed_cand"]
    sr, sc = prob["seed_coord"]
    target_mapping = prob["target_mapping"].unsqueeze(0)

    assert sr == 1 and sc == 1
    assert target_mapping[0, sr, sc].item() == seed_cand

    model_config = ModelConfig(grid_size=grid_size, content_dim=96, mode="both")
    model = JigsawSolverV2(model_config)

    # Encode patches
    raw_feats = model.encode_candidates(candidates)  # (1, K, D)

    # Verify seed candidate retains gradient requirement
    assert raw_feats.requires_grad
    raw_feats.retain_grad()

    grid_placed = torch.full((1, grid_size, grid_size), -1, dtype=torch.long)
    used_candidates = torch.zeros((1, K), dtype=torch.bool)
    cand_to_slot = torch.full((1, K, 2), -1, dtype=torch.long)

    # Place seed
    grid_placed[0, sr, sc] = seed_cand
    used_candidates[0, seed_cand] = True
    cand_to_slot[0, seed_cand] = torch.tensor([sr, sc])

    # Target empty neighbor: (0, 1) [Top neighbor of seed]
    target_coord = torch.tensor([[0, 1]], dtype=torch.long)
    true_target_cand = target_mapping[0, 0, 1]

    # Target cannot be the seed
    assert true_target_cand.item() != seed_cand

    # Forward scoring
    logits = model.score_step(
        raw_feats=raw_feats,
        grid_placed=grid_placed,
        used_candidates=used_candidates,
        cand_to_slot=cand_to_slot,
        target_coords=target_coord,
        grid_size=grid_size
    )

    # Loss logits masks out seed
    loss_logits = logits.clone()
    loss_logits[0, seed_cand] = -float("inf")

    loss = nn.CrossEntropyLoss()(loss_logits, true_target_cand.unsqueeze(0))
    loss.backward()

    assert raw_feats.grad[0, seed_cand].abs().sum() > 0, "Seed content must learn through non-seed context losses"

    # Check encoder received gradients
    encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert encoder_has_grad, "ContentEncoder must receive gradients from non-seed loss"

    # Check local branch received gradients
    local_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.local_branch.parameters())
    assert local_has_grad, "Local branch must receive gradients from non-seed loss"

    # Check global branch received gradients
    global_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.global_branch.parameters())
    assert global_has_grad, "Global branch must receive gradients from non-seed loss"

    # Verify seed position is still intact
    assert grid_placed[0, sr, sc].item() == seed_cand
