"""
Unit tests for deterministic BFS order, candidate non-repetition, and autonomous solver.
"""

import torch
import pytest
from v3.config import ModelConfig
from v3.model import JigsawSolverV3
from v3.dataset import generate_bfs_order, assemble_patches, pad_and_slice_image
from v3.solver import solve_batch, compute_puzzle_accuracy


def test_bfs_order_completeness():
    grid_size = 5
    seed_coord = (2, 2)
    order = generate_bfs_order(grid_size, seed_coord)

    assert len(order) == 24
    assert seed_coord not in order
    assert len(set(order)) == 24
    for r, c in order:
        assert 0 <= r < grid_size and 0 <= c < grid_size


def test_autonomous_solver_non_repetition():
    model = JigsawSolverV3(ModelConfig())
    model.eval()

    B = 3
    patches = torch.randn(B, 25, 3, 13, 13)
    seed_cands = torch.tensor([12, 5, 20])
    seed_coords = torch.tensor([[2, 2], [2, 2], [2, 2]])

    out = solve_batch(
        model=model,
        patches=patches,
        seed_cands=seed_cands,
        seed_coords=seed_coords,
        grid_size=5,
        allow_repeats=False
    )

    grid_placed = out["grid_placed"]
    assert grid_placed.shape == (B, 5, 5)

    # Verify every candidate is used exactly once per sample (no duplicates)
    for b in range(B):
        placed_indices = grid_placed[b].flatten().tolist()
        assert len(placed_indices) == 25
        assert set(placed_indices) == set(range(25)), "Candidates were duplicated or missing!"


def test_autonomous_solver_preserves_training_mode():
    model = JigsawSolverV3(ModelConfig(
        content_dim=32,
        local_layers=1,
        local_heads=4,
        global_layers=1,
        global_heads=4,
        global_ffn_dim=64,
        proj_hidden_dim=64,
        proj_out_dim=64,
        cnn_stages=(8, 16, 24),
        gradient_checkpointing=True
    ))
    model.train()
    out = solve_batch(
        model=model,
        patches=torch.randn(1, 25, 3, 13, 13),
        seed_cands=torch.tensor([12]),
        seed_coords=torch.tensor([[2, 2]]),
        grid_size=5
    )

    assert out["grid_placed"].shape == (1, 5, 5)
    assert model.training, "Inference helper must not change the caller's model mode"


def test_puzzle_accuracy_calculation():
    grid_placed = torch.arange(25).view(1, 5, 5)
    target_mapping = torch.arange(25).view(1, 5, 5)
    seed_coords = torch.tensor([[2, 2]])

    patch_acc, perfect = compute_puzzle_accuracy(grid_placed, target_mapping, seed_coords, grid_size=5)
    assert patch_acc.item() == 1.0
    assert perfect.item() == 1.0

    # Perturb 1 slot
    grid_placed_wrong = grid_placed.clone()
    grid_placed_wrong[0, 0, 0] = 99
    patch_acc_wrong, perfect_wrong = compute_puzzle_accuracy(grid_placed_wrong, target_mapping, seed_coords, grid_size=5)
    assert patch_acc_wrong.item() == pytest.approx(23.0 / 24.0, abs=1e-5)
    assert perfect_wrong.item() == 0.0
