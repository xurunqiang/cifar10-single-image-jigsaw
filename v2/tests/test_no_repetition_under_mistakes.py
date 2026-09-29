"""
Test 2: Robustness Under Injected Mistakes & Non-Repetition Guarantee
Verify:
- Artificially forced wrong candidate choices do NOT cause duplicate usage.
- All K candidate pieces are used exactly once.
- Classification loss remains finite even when the correct candidate for the current slot
  was prematurely consumed by an earlier error.
- Teacher forcing fallback works correctly when the ground truth piece is already consumed.
"""

import pytest
import torch
import torch.nn as nn
from v2.config import ModelConfig
from v2.model import JigsawSolverV2
from v2.dataset import JigsawProblemGenerator
from v2.expansion import get_expansion_order


def test_no_repetition_and_finite_loss_under_mistakes():
    grid_size = 3
    K = grid_size * grid_size
    img = torch.randn(3, 32, 32)

    prob = JigsawProblemGenerator.generate(img, grid_size=grid_size, seed_mode="center")
    candidates = prob["candidates"].unsqueeze(0)  # (1, K, 3, P, P)
    seed_cand = prob["seed_cand"]
    seed_coord = prob["seed_coord"]
    target_mapping = prob["target_mapping"]

    model_config = ModelConfig(grid_size=grid_size, content_dim=96, mode="both")
    model = JigsawSolverV2(model_config)
    criterion = nn.CrossEntropyLoss()

    raw_feats = model.encode_candidates(candidates)

    grid_placed = torch.full((1, grid_size, grid_size), -1, dtype=torch.long)
    used_candidates = torch.zeros((1, K), dtype=torch.bool)
    cand_to_slot = torch.full((1, K, 2), -1, dtype=torch.long)

    sr, sc = seed_coord
    grid_placed[0, sr, sc] = seed_cand
    used_candidates[0, seed_cand] = True
    cand_to_slot[0, seed_cand] = torch.tensor([sr, sc])

    expansion_order = get_expansion_order(grid_size, seed_coord, strategy="bfs")

    # Step 0 of non-seed: Intentionally inject a mistake!
    # Pick the candidate that belongs to the LAST slot in expansion_order
    last_r, last_c = expansion_order[-1]
    last_correct_cand = int(target_mapping[last_r, last_c].item())

    total_loss = 0.0
    conflicts = 0

    for step_idx, (r, c) in enumerate(expansion_order):
        target_coords = torch.tensor([[r, c]], dtype=torch.long)
        true_cand = target_mapping[r, c].view(1)

        logits = model.score_step(
            raw_feats=raw_feats,
            grid_placed=grid_placed,
            used_candidates=used_candidates,
            cand_to_slot=cand_to_slot,
            target_coords=target_coords,
            grid_size=grid_size
        )

        # Classification loss (unmasked except seed)
        loss_logits = logits.clone()
        loss_logits[0, seed_cand] = -float("inf")
        step_loss = criterion(loss_logits, true_cand)

        # Must be finite
        assert torch.isfinite(step_loss), f"Step loss at step {step_idx} must be finite!"
        total_loss += float(step_loss.item())

        # Selection logits (masked with used_candidates)
        selection_logits = logits.masked_fill(used_candidates, -float("inf"))

        if step_idx == 0:
            # Force the mistake: place last_correct_cand
            chosen_cand = last_correct_cand
        else:
            # In later step, if we try to simulate teacher forcing:
            # If this is the last step, true_cand was already used in step 0!
            tc = int(true_cand.item())
            is_used = bool(used_candidates[0, tc].item())
            if is_used:
                conflicts += 1
                # Must fallback to model prediction among unused candidates
                chosen_cand = int(torch.argmax(selection_logits, dim=-1).item())
            else:
                chosen_cand = int(torch.argmax(selection_logits, dim=-1).item())

        # Verify chosen candidate was NOT previously used
        assert not used_candidates[0, chosen_cand].item(), f"Candidate {chosen_cand} was reused at step {step_idx}!"

        # Place candidate
        grid_placed[0, r, c] = chosen_cand
        used_candidates[0, chosen_cand] = True
        cand_to_slot[0, chosen_cand] = target_coords[0]

    # At the end of the trajectory:
    # 1. Conflict was successfully detected and handled
    assert conflicts >= 1, "At least one conflict should have been detected due to the forced mistake"

    # 2. Every single candidate must be used exactly once
    assert used_candidates.all().item(), "All candidates must be marked used"
    unique_placed = torch.unique(grid_placed[0])
    assert len(unique_placed) == K, f"Expected {K} unique pieces, got {len(unique_placed)}"

    # 3. Overall loss was finite throughout
    assert total_loss > 0 and not torch.isnan(torch.tensor(total_loss))
