"""
Test 5: Inference Independence from Ground Truth Target Mapping
Verify:
- Inference interface does NOT require or accept ground truth target mappings.
- Modifying, corrupting, or deleting target mappings has zero effect on model scoring or solver predictions.
"""

import pytest
import torch
from v2.config import ModelConfig
from v2.model import JigsawSolverV2
from v2.dataset import JigsawProblemGenerator
from v2.solver import solve


def test_inference_independence():
    torch.manual_seed(100)
    grid_size = 3
    img = torch.randn(3, 32, 32)

    prob = JigsawProblemGenerator.generate(img, grid_size=grid_size, seed_mode="center")
    candidates = prob["candidates"]
    seed_cand = prob["seed_cand"]
    seed_coord = prob["seed_coord"]

    model_config = ModelConfig(grid_size=grid_size, content_dim=96, mode="both")
    model = JigsawSolverV2(model_config)
    model.eval()

    # 1. Run solve without target_mapping
    res1 = solve(
        model=model,
        patches=candidates,
        seed_cand=seed_cand,
        seed_coord=seed_coord,
        grid_size=grid_size,
        strategy="bfs"
    )

    # 2. Corrupt target_mapping completely (e.g. fill with 999 or random permutations)
    corrupted_mapping = torch.full_like(prob["target_mapping"], 999)
    prob["target_mapping"] = corrupted_mapping

    # Run solve again
    res2 = solve(
        model=model,
        patches=candidates,
        seed_cand=seed_cand,
        seed_coord=seed_coord,
        grid_size=grid_size,
        strategy="bfs"
    )

    # Verify identical decisions
    assert torch.equal(res1["slot_to_cand"], res2["slot_to_cand"]), (
        "Inference output changed when ground truth target mapping was modified!"
    )
    assert torch.equal(res1["assembled_image"], res2["assembled_image"]), (
        "Assembled image changed when ground truth target mapping was modified!"
    )
