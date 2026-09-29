"""
Autonomous Solver and Puzzle Reconstructor for v2:
- Fully autonomous assembly without ground-truth labels
- Enforces candidate non-repetition via used_candidates masking
- Reconstructs assembled images and records step-by-step trajectories
"""

import random
from typing import Dict, Any, List, Tuple, Optional
import torch

from .expansion import get_expansion_order
from .dataset import assemble_patches


def validate_inputs(patches, seed_cands, seed_coords, grid_size):
    B, K = patches.shape[:2]
    if K != grid_size ** 2 or seed_cands.shape != (B,) or seed_coords.shape != (B, 2):
        raise ValueError("Candidate count or seed shapes do not match puzzle grid")
    if ((seed_cands < 0) | (seed_cands >= K)).any():
        raise ValueError("Seed candidate index out of bounds")
    if ((seed_coords < 0) | (seed_coords >= grid_size)).any():
        raise ValueError("Seed coordinate out of bounds")


@torch.no_grad()
def solve(
    model: torch.nn.Module,
    patches: torch.Tensor,
    seed_cand: int,
    seed_coord: Tuple[int, int],
    grid_size: int,
    strategy: str = "random_frontier",
    rng: Optional[random.Random] = None,
    allow_repeats: bool = False,
) -> Dict[str, Any]:
    """
    Solve a single jigsaw puzzle instance autonomously.
    Args:
        model: JigsawSolverV2 model
        patches: (K, 3, P, P)
        seed_cand: int
        seed_coord: (sr, sc)
        grid_size: int
        strategy: "random_frontier" or "bfs"
        allow_repeats: If True, do not mask candidates already placed. Intended
            for inference ablations; the default solver still guarantees unique use.
        rng: optional random.Random
    Returns:
        dict containing:
        - slot_to_cand: (grid_size, grid_size) tensor
        - assembled_image: (3, H, W) tensor
        - trajectory: list of step details
    """
    model.eval()
    device = next(model.parameters()).device
    patches_dev = patches.unsqueeze(0).to(device)  # (1, K, 3, P, P)
    K = grid_size * grid_size
    validate_inputs(patches_dev, torch.tensor([seed_cand]), torch.tensor([seed_coord]), grid_size)

    # Encode patches once
    raw_feats = model.encode_candidates(patches_dev)  # (1, K, D)

    # Initialize state
    grid_placed = torch.full((1, grid_size, grid_size), -1, dtype=torch.long, device=device)
    used_candidates = torch.zeros((1, K), dtype=torch.bool, device=device)
    cand_to_slot = torch.full((1, K, 2), -1, dtype=torch.long, device=device)

    sr, sc = seed_coord
    grid_placed[0, sr, sc] = seed_cand
    used_candidates[0, seed_cand] = True
    cand_to_slot[0, seed_cand] = torch.tensor([sr, sc], device=device)

    # Expansion order
    expansion_order = get_expansion_order(grid_size, seed_coord, strategy=strategy, rng=rng)

    trajectory: List[Dict[str, Any]] = []

    for step_idx, (r, c) in enumerate(expansion_order):
        target_coords = torch.tensor([[r, c]], dtype=torch.long, device=device)

        # Model step scoring
        logits = model.score_step(
            raw_feats=raw_feats,
            grid_placed=grid_placed,
            used_candidates=used_candidates,
            cand_to_slot=cand_to_slot,
            target_coords=target_coords,
            grid_size=grid_size
        )  # (1, K)

        # By default strictly prohibit already used candidates. The optional
        # ablation lets us measure inference without this uniqueness constraint.
        selection_logits = logits if allow_repeats else logits.masked_fill(used_candidates, -float("inf"))

        # Greedy selection
        chosen_cand = int(torch.argmax(selection_logits, dim=-1).item())
        already_used = bool(used_candidates[0, chosen_cand].item())

        # Update placement
        grid_placed[0, r, c] = chosen_cand
        used_candidates[0, chosen_cand] = True
        cand_to_slot[0, chosen_cand] = target_coords[0]

        trajectory.append({
            "step": step_idx,
            "target_slot": (r, c),
            "chosen_cand": chosen_cand,
            "planned_prompt": False,
            "actual_prompt": False,
            "already_used": already_used,
        })

    # Assemble reconstructed image
    assembled = assemble_patches(patches_dev[0], grid_placed[0], grid_size=grid_size)

    return {
        "slot_to_cand": grid_placed[0].cpu(),
        "assembled_image": assembled.cpu(),
        "trajectory": trajectory,
    }


@torch.no_grad()
def solve_batch(
    model: torch.nn.Module,
    patches: torch.Tensor,
    seed_cands: torch.Tensor,
    seed_coords: torch.Tensor,
    grid_size: int,
    strategy: str = "random_frontier",
    rng_seeds: Optional[List[int]] = None,
    record_trajectories: bool = True,
    allow_repeats: bool = False,
) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
    """
    Batched autonomous solver.
    Args:
        patches: (B, K, 3, P, P)
        seed_cands: (B,)
        seed_coords: (B, 2)
        grid_size: int
        strategy: "random_frontier" or "bfs"
        allow_repeats: If True, do not mask candidates already placed. Intended
            for inference ablations; the default solver still guarantees unique use.
    Returns:
        grid_placed: (B, grid_size, grid_size)
        trajectories: list of trajectories for each sample in batch
    """
    model.eval()
    device = next(model.parameters()).device
    B, K, _, _, _ = patches.shape
    patches_dev = patches.to(device)
    seed_cands_dev = seed_cands.to(device)
    seed_coords_dev = seed_coords.to(device)
    validate_inputs(patches_dev, seed_cands_dev, seed_coords_dev, grid_size)
    if rng_seeds is not None and len(rng_seeds) != B:
        raise ValueError("One expansion RNG seed is required per sample")

    raw_feats = model.encode_candidates(patches_dev)  # (B, K, D)

    grid_placed = torch.full((B, grid_size, grid_size), -1, dtype=torch.long, device=device)
    used_candidates = torch.zeros((B, K), dtype=torch.bool, device=device)
    cand_to_slot = torch.full((B, K, 2), -1, dtype=torch.long, device=device)

    # Initialize seed
    batch_idx = torch.arange(B, device=device)
    grid_placed[batch_idx, seed_coords_dev[:, 0], seed_coords_dev[:, 1]] = seed_cands_dev
    used_candidates[batch_idx, seed_cands_dev] = True
    cand_to_slot[batch_idx, seed_cands_dev] = seed_coords_dev

    # Expansion order per sample
    expansion_orders: List[List[Tuple[int, int]]] = []
    for b in range(B):
        coord = (int(seed_coords[b, 0].item()), int(seed_coords[b, 1].item()))
        rng = random.Random(rng_seeds[b]) if rng_seeds is not None else None
        order = get_expansion_order(grid_size, coord, strategy=strategy, rng=rng)
        expansion_orders.append(order)

    trajectories: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
    num_steps = K - 1
    expansion_tensor = torch.tensor(expansion_orders, dtype=torch.long, device=device)

    for step_idx in range(num_steps):
        target_coords_list = [expansion_orders[b][step_idx] for b in range(B)]
        target_coords = expansion_tensor[:, step_idx]

        logits = model.score_step(
            raw_feats=raw_feats,
            grid_placed=grid_placed,
            used_candidates=used_candidates,
            cand_to_slot=cand_to_slot,
            target_coords=target_coords,
            grid_size=grid_size
        )  # (B, K)

        selection_logits = logits if allow_repeats else logits.masked_fill(used_candidates, -float("inf"))
        chosen_cands = torch.argmax(selection_logits, dim=-1)  # (B,)
        already_used = used_candidates[batch_idx, chosen_cands] if record_trajectories else None

        grid_placed[batch_idx, target_coords[:, 0], target_coords[:, 1]] = chosen_cands
        used_candidates[batch_idx, chosen_cands] = True
        cand_to_slot[batch_idx, chosen_cands] = target_coords

        chosen_cpu = chosen_cands.tolist() if record_trajectories else []
        already_used_cpu = already_used.tolist() if record_trajectories else []
        for b in range(B) if record_trajectories else []:
            trajectories[b].append({
                "step": step_idx,
                "target_slot": target_coords_list[b],
                "chosen_cand": chosen_cpu[b],
                "planned_prompt": False,
                "actual_prompt": False,
                "already_used": already_used_cpu[b],
            })

    return grid_placed.cpu(), trajectories
