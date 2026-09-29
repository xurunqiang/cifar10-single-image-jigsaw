"""
Test 4: Permutation Equivariance
Verify:
- Permuting the input candidate pool and updating the seed candidate index synchronously
  results in the identical physical assembled image under deterministic decoding.
"""

import pytest
import random
import torch
from v2.config import ModelConfig
from v2.model import JigsawSolverV2
from v2.dataset import pad_and_slice_image
from v2.solver import solve


def test_permutation_equivariance():
    torch.manual_seed(42)
    random.seed(42)

    grid_size = 3
    K = grid_size * grid_size
    img = torch.randn(3, 32, 32)
    orig_patches, _ = pad_and_slice_image(img, grid_size)  # (K, 3, P, P)

    # Initial candidate shuffle 1
    perm1 = list(range(K))
    random.shuffle(perm1)
    cand1 = orig_patches[perm1]
    seed_slot = (1, 1)
    seed_raster = 1 * grid_size + 1
    seed_cand1 = perm1.index(seed_slot[0] * grid_size + seed_slot[1])

    # Model
    model_config = ModelConfig(grid_size=grid_size, content_dim=96, mode="both")
    model = JigsawSolverV2(model_config)
    model.eval()

    # Solve puzzle 1 with BFS (deterministic order)
    res1 = solve(
        model=model,
        patches=cand1,
        seed_cand=seed_cand1,
        seed_coord=seed_slot,
        grid_size=grid_size,
        strategy="bfs"
    )

    # Create a secondary permutation sigma of the candidates
    sigma = list(range(K))
    random.shuffle(sigma)
    cand2 = cand1[sigma]  # cand2[j] = cand1[sigma[j]]
    # Find new seed candidate index in cand2
    seed_cand2 = sigma.index(seed_cand1)

    # Solve puzzle 2 with identical seed coordinate and BFS
    res2 = solve(
        model=model,
        patches=cand2,
        seed_cand=seed_cand2,
        seed_coord=seed_slot,
        grid_size=grid_size,
        strategy="bfs"
    )

    # Verify physical reconstructed images are identical
    assembled1 = res1["assembled_image"]
    assembled2 = res2["assembled_image"]
    max_diff = (assembled1 - assembled2).abs().max().item()

    assert max_diff < 1e-5, f"Assembled images differ! Max diff: {max_diff}"

    # Also verify that for each slot (r, c), the physical patch placed is identical
    slot_to_cand1 = res1["slot_to_cand"]
    slot_to_cand2 = res2["slot_to_cand"]

    for r in range(grid_size):
        for c in range(grid_size):
            c1 = int(slot_to_cand1[r, c].item())
            c2 = int(slot_to_cand2[r, c].item())
            # Patch in cand1[c1] must equal patch in cand2[c2]
            p1 = cand1[c1]
            p2 = cand2[c2]
            patch_diff = (p1 - p2).abs().max().item()
            assert patch_diff < 1e-5, f"Physical patch at ({r}, {c}) differs! Diff: {patch_diff}"
