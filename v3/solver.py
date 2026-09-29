"""
Autonomous Solver and Puzzle Reconstructor for v3:
- Fully autonomous assembly without ground-truth labels
- Enforces candidate non-repetition via used_candidates masking
- Computes coordinate-free Virtual Patch Z_virtual on final placed board
- Evaluates non-seed patch accuracy and perfect assembly rate
"""

from typing import Dict, Any, List, Tuple, Optional
import torch

from .dataset import generate_bfs_order, assemble_patches


def validate_inputs(
    patches: torch.Tensor,
    seed_cands: torch.Tensor,
    seed_coords: torch.Tensor,
    grid_size: int
) -> None:
    B, K = patches.shape[:2]
    if K != grid_size ** 2 or seed_cands.shape != (B,) or seed_coords.shape != (B, 2):
        raise ValueError(f"Candidate count ({K}) or seed shapes do not match {grid_size}x{grid_size} puzzle grid")
    if ((seed_cands < 0) | (seed_cands >= K)).any():
        raise ValueError("Seed candidate index out of bounds")
    if ((seed_coords < 0) | (seed_coords >= grid_size)).any():
        raise ValueError("Seed coordinate out of bounds")


def compute_puzzle_accuracy(
    grid_placed: torch.Tensor,       # (B, G, G)
    target_mapping: torch.Tensor,    # (B, G, G)
    seed_coords: torch.Tensor,       # (B, 2)
    grid_size: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes patch accuracy (excluding seed) and perfect puzzle accuracy.
    Returns:
        patch_acc: (B,) float tensor, accuracy over the 24 non-seed positions
        perfect: (B,) float tensor, 1.0 if all 24 non-seed positions match, else 0.0
    """
    B = grid_placed.shape[0]
    device = grid_placed.device

    # Create mask for non-seed positions
    non_seed_mask = torch.ones((B, grid_size, grid_size), dtype=torch.bool, device=device)
    batch_idx = torch.arange(B, device=device)
    non_seed_mask[batch_idx, seed_coords[:, 0], seed_coords[:, 1]] = False

    matches = (grid_placed == target_mapping) & non_seed_mask
    num_non_seed = grid_size * grid_size - 1

    correct_counts = matches.sum(dim=(1, 2)).float()
    patch_acc = correct_counts / float(num_non_seed)
    perfect = (correct_counts == num_non_seed).float()

    return patch_acc, perfect


def compute_neighbor_accuracy(grid_placed: torch.Tensor, target_mapping: torch.Tensor) -> torch.Tensor:
    """Fraction of horizontal/vertical edges preserving the true relative offsets."""
    B, G, _ = grid_placed.shape
    true_slots = target_mapping.flatten(1).argsort(1).gather(1, grid_placed.flatten(1)).reshape(B, G, G)
    rows, cols = true_slots // G, true_slots % G
    horizontal = (rows[:, :, 1:] == rows[:, :, :-1]) & (cols[:, :, 1:] == cols[:, :, :-1] + 1)
    vertical = (rows[:, 1:, :] == rows[:, :-1, :]) & (cols[:, 1:, :] == cols[:, :-1, :] + 1)
    return (horizontal.sum((1, 2)) + vertical.sum((1, 2))).float() / (2 * G * (G - 1))


@torch.no_grad()
def solve_batch(
    model: torch.nn.Module,
    patches: torch.Tensor,
    seed_cands: torch.Tensor,
    seed_coords: torch.Tensor,
    grid_size: int = 5,
    expansion_order: Optional[List[Tuple[int, int]]] = None,
    allow_repeats: bool = False,
    raw_feats: Optional[torch.Tensor] = None,
    compute_representation: bool = True
) -> Dict[str, Any]:
    """
    Batched autonomous solver for v3.
    Args:
        model: JigsawSolverV3
        patches: (B, K, 3, 13, 13)
        seed_cands: (B,)
        seed_coords: (B, 2)
        grid_size: int
        expansion_order: sequence of (r, c) slots to fill; defaults to BFS from seed
        allow_repeats: if True, disables non-repetition masking (for ablations)
        raw_feats: optional precomputed candidate features, useful when a caller
            already encoded the same patches for a differentiable training loss.
    Returns:
        Dict containing:
            grid_placed: (B, G, G) placed candidate indices
            cand_to_slot: (B, K, 2) assigned coordinates for each candidate
            z_virtual: (B, D) coordinate-free whole-image representation
            attn_weights: (B, K) attention weights over placed candidates
            raw_feats: (B, K, D) candidate features from CNN encoder
    """
    device = next(model.parameters()).device
    B, K, _, _, _ = patches.shape

    patches_dev = patches.to(device)
    seed_cands_dev = seed_cands.to(device)
    seed_coords_dev = seed_coords.to(device)
    validate_inputs(patches_dev, seed_cands_dev, seed_coords_dev, grid_size)

    # One shared expansion order is only valid when every sample has the same seed.
    if not torch.equal(seed_coords_dev, seed_coords_dev[:1].expand_as(seed_coords_dev)):
        raise ValueError("All puzzles in a batch must use the same seed coordinate")

    seed_coord = tuple(int(x) for x in seed_coords_dev[0].tolist())
    if expansion_order is None:
        expansion_order = generate_bfs_order(grid_size, seed_coord)
    else:
        expansion_order = [tuple(map(int, coord)) for coord in expansion_order]
        expected = {
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if (r, c) != seed_coord
        }
        if len(expansion_order) != len(expected) or set(expansion_order) != expected:
            raise ValueError("expansion_order must contain each non-seed grid slot exactly once")

    # Encode patches once
    if raw_feats is None:
        raw_feats = model.encode_candidates(patches_dev)  # (B, K, D)
    else:
        raw_feats = raw_feats.to(device)
        if raw_feats.shape[:2] != (B, K):
            raise ValueError(f"raw_feats must start with shape {(B, K)}, got {tuple(raw_feats.shape)}")

    # Initialize state
    grid_placed = torch.full((B, grid_size, grid_size), -1, dtype=torch.long, device=device)
    used_candidates = torch.zeros((B, K), dtype=torch.bool, device=device)
    cand_to_slot = torch.full((B, K, 2), -1, dtype=torch.long, device=device)

    # Place seed
    batch_idx = torch.arange(B, device=device)
    grid_placed[batch_idx, seed_coords_dev[:, 0], seed_coords_dev[:, 1]] = seed_cands_dev
    used_candidates[batch_idx, seed_cands_dev] = True
    cand_to_slot[batch_idx, seed_cands_dev] = seed_coords_dev

    # Sequential greedy placement along BFS expansion order
    for r, c in expansion_order:
        target_coords = torch.tensor([[r, c]], dtype=torch.long, device=device).expand(B, 2)

        scores, _ = model.score_step(
            raw_feats=raw_feats,
            grid_placed=grid_placed,
            used_candidates=used_candidates,
            cand_to_slot=cand_to_slot,
            target_coords=target_coords,
            grid_size=grid_size
        )  # (B, K)

        if not allow_repeats:
            scores = scores.masked_fill(used_candidates, -float("inf"))

        chosen_cand = torch.argmax(scores, dim=-1)  # (B,)

        # Update board and state
        grid_placed[batch_idx, r, c] = chosen_cand
        used_candidates[batch_idx, chosen_cand] = True
        cand_to_slot[batch_idx, chosen_cand] = target_coords

    # No final representation is needed for the training-only autonomous rollout.
    z_virtual = attn_weights = None
    if compute_representation:
        if allow_repeats:
            raise ValueError("Virtual board features require a permutation without repeated candidates")
        z_virtual, attn_weights = model.compute_virtual_patch(
            raw_feats=raw_feats, cand_to_slot=cand_to_slot, used_candidates=used_candidates
        )

    return {
        "grid_placed": grid_placed,
        "cand_to_slot": cand_to_slot,
        "z_virtual": z_virtual,
        "attn_weights": attn_weights,
        "raw_feats": raw_feats,
    }


@torch.no_grad()
def solve_single(
    model: torch.nn.Module,
    patches: torch.Tensor,
    seed_cand: int,
    seed_coord: Optional[Tuple[int, int]] = None,
    grid_size: int = 5,
    target_mapping: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Solves a single jigsaw puzzle and optionally reconstructs the full image.
    """
    seed_coord = seed_coord or (grid_size // 2, grid_size // 2)
    patches_b = patches.unsqueeze(0)  # (1, K, 3, 13, 13)
    seed_cands_b = torch.tensor([seed_cand], dtype=torch.long)
    seed_coords_b = torch.tensor([seed_coord], dtype=torch.long)

    out = solve_batch(
        model=model,
        patches=patches_b,
        seed_cands=seed_cands_b,
        seed_coords=seed_coords_b,
        grid_size=grid_size
    )

    grid_placed = out["grid_placed"][0].cpu()
    assembled = assemble_patches(patches.cpu(), grid_placed, grid_size=grid_size)

    result = {
        "grid_placed": grid_placed,
        "assembled_image": assembled,
        "z_virtual": out["z_virtual"][0].cpu(),
        "attn_weights": out["attn_weights"][0].cpu(),
        "raw_feats": out["raw_feats"][0].cpu(),
    }

    if target_mapping is not None:
        patch_acc, perfect = compute_puzzle_accuracy(
            grid_placed.unsqueeze(0),
            target_mapping.unsqueeze(0),
            seed_coords_b.cpu(),
            grid_size=grid_size
        )
        result["patch_acc"] = patch_acc.item()
        result["perfect"] = perfect.item()

    return result
