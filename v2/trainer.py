"""
Training Engine for v2:
- Disentangled masking (classification loss on non-seed candidates vs selection mask on unused candidates)
- Curriculum-driven teacher forcing decay with fallback handling
- Comprehensive metric tracking (planned prompt, actual prompt, conflict ratio, autonomous vs prompted accuracy)
- Deterministic validation and checkpointing (best.pt and latest.pt)
"""

import os
import random
import tempfile
import warnings
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .model import JigsawSolverV2
from .expansion import get_expansion_order
from .curriculum import CurriculumScheduler, build_lr_scheduler
from .solver import solve_batch
from .dataset import set_dataset_epoch


def compute_pairwise_accuracy(
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    grid_size: int
) -> float:
    """
    Computes fraction of adjacent edges (horizontal and vertical) that are correct.
    pred_grid: (B, G, G)
    target_grid: (B, G, G)
    """
    # Compare relative true positions, so a correct pair counts even when translated.
    B, G, _ = pred_grid.shape
    original_slots = target_grid.flatten(1).argsort(dim=1).gather(1, pred_grid.flatten(1)).view(B, G, G)
    rows, cols = original_slots // G, original_slots % G
    horizontal = (rows[:, :, 1:] == rows[:, :, :-1]) & (cols[:, :, 1:] == cols[:, :, :-1] + 1)
    vertical = (rows[:, 1:, :] == rows[:, :-1, :] + 1) & (cols[:, 1:, :] == cols[:, :-1, :])
    return float((horizontal.sum() + vertical.sum()).item()) / max(1, B * 2 * G * (G - 1))


def save_atomic(state, path):
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".tmp", delete=False) as handle:
        temporary = handle.name
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TrainerV2:
    def __init__(
        self,
        model: JigsawSolverV2,
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        self.model = model.to(device)
        self.model_config = model_config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grid_size = model_config.grid_size
        self.K = self.grid_size * self.grid_size

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay
        )

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_epochs=train_config.warmup_epochs,
            total_epochs=train_config.epochs
        )

        self.curriculum = CurriculumScheduler(total_epochs=train_config.epochs)
        self.criterion = nn.CrossEntropyLoss()

        self.start_epoch = 1
        self.best_acc = -1.0
        self.best_perfect_acc = -1.0

        self.save_dir = train_config.get_save_dir(self.grid_size)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        stage_info = self.curriculum.get_stage_info(epoch)
        p_teacher = stage_info["teacher_forcing_prob"]

        # If dataset supports epoch updates (for dynamic random seed selection)
        set_dataset_epoch(self.train_loader.dataset, epoch)

        total_loss = 0.0
        total_steps = 0
        planned_prompts = 0
        actual_prompts = 0
        conflicts = 0
        targets_already_used = 0
        total_samples = 0
        model_correct_selections = 0
        prompted_correct_placements = 0

        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        epoch_lr = self.optimizer.param_groups[0]["lr"]

        for batch in self.train_loader:
            candidates = batch["candidates"].to(self.device)         # (B, K, 3, P, P)
            seed_cands = batch["seed_cand"].to(self.device)           # (B,)
            seed_coords = batch["seed_coord"].to(self.device)         # (B, 2)
            target_mappings = batch["target_mapping"].to(self.device) # (B, G, G)

            B = candidates.shape[0]
            total_samples += B
            batch_idx = torch.arange(B, device=self.device)
            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Encode all candidate patches once per batch
                raw_feats = self.model.encode_candidates(candidates)  # (B, K, D)

                # Initialize grid and candidate tracking
                grid_placed = torch.full((B, self.grid_size, self.grid_size), -1, dtype=torch.long, device=self.device)
                used_candidates = torch.zeros((B, self.K), dtype=torch.bool, device=self.device)
                cand_to_slot = torch.full((B, self.K, 2), -1, dtype=torch.long, device=self.device)

                # Place seed
                grid_placed[batch_idx, seed_coords[:, 0], seed_coords[:, 1]] = seed_cands
                used_candidates[batch_idx, seed_cands] = True
                cand_to_slot[batch_idx, seed_cands] = seed_coords

                # Determine expansion trajectory for each sample independently
                orders_list = []
                for b in range(B):
                    coord = (int(seed_coords[b, 0].item()), int(seed_coords[b, 1].item()))
                    order = get_expansion_order(
                        self.grid_size,
                        coord,
                        strategy=self.train_config.expansion_strategy,
                        rng=random.Random(batch["expansion_seeds"][b]) if "expansion_seeds" in batch else None,
                    )
                    orders_list.append(order)
                expansion_tensor = torch.tensor(orders_list, dtype=torch.long, device=self.device)  # (B, K-1, 2)

                batch_loss = torch.tensor(0.0, device=self.device)
                num_non_seed_steps = self.K - 1

                for step_idx in range(num_non_seed_steps):
                    target_coords = expansion_tensor[:, step_idx]  # (B, 2)

                    # Ground truth correct candidates for this slot
                    true_cands = target_mappings[batch_idx, target_coords[:, 0], target_coords[:, 1]]  # (B,)

                    # Step scoring: does NOT receive target_mapping
                    logits = self.model.score_step(
                        raw_feats=raw_feats,
                        grid_placed=grid_placed,
                        used_candidates=used_candidates,
                        cand_to_slot=cand_to_slot,
                        target_coords=target_coords,
                        grid_size=self.grid_size
                    )  # (B, K)

                    # --- Classification Loss (Evaluated on all non-seed candidates) ---
                    # Mask out seed candidate so it is never chosen as the target of another slot
                    loss_logits = logits.clone()
                    loss_logits[batch_idx, seed_cands] = -float("inf")

                    step_loss = self.criterion(loss_logits, true_cands)
                    batch_loss = batch_loss + step_loss

                    # --- Candidate Selection (Evaluated ONLY on unused candidates) ---
                    selection_logits = logits.masked_fill(used_candidates, -float("inf"))
                    pred_cands = torch.argmax(selection_logits, dim=-1)  # (B,)

                    # Track autonomous accuracy of the model before prompt override
                    model_correct = (pred_cands == true_cands)
                    model_correct_selections += model_correct.detach().sum()

                    # --- Curriculum Teacher Forcing Execution (Vectorized) ---
                    u = torch.rand(B, device=self.device)
                    prompt_draw = (u < p_teacher)
                    tc_used = used_candidates[batch_idx, true_cands]
                    targets_already_used += tc_used.detach().sum()
                    can_prompt = prompt_draw & (~tc_used)
                    conflicts_mask = prompt_draw & tc_used

                    planned_prompts += prompt_draw.detach().sum()
                    actual_prompts += can_prompt.detach().sum()
                    conflicts += conflicts_mask.detach().sum()

                    actual_placed = torch.where(can_prompt, true_cands, pred_cands)

                    prompted_correct = (actual_placed == true_cands)
                    prompted_correct_placements += prompted_correct.detach().sum()

                    # Update placements and used state out-of-place to preserve autograd graph history
                    grid_placed = grid_placed.clone()
                    grid_placed[batch_idx, target_coords[:, 0], target_coords[:, 1]] = actual_placed

                    used_candidates = used_candidates.clone()
                    used_candidates[batch_idx, actual_placed] = True

                    cand_to_slot = cand_to_slot.clone()
                    cand_to_slot[batch_idx, actual_placed] = target_coords

                    total_steps += B

                # Average loss over all non-seed steps
                batch_loss = batch_loss / float(num_non_seed_steps)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
            self.optimizer.step()

            total_loss += float(batch_loss.item()) * B

        if not total_samples:
            raise ValueError("Training loader is empty")
        (planned_prompts, actual_prompts, conflicts, targets_already_used,
         model_correct_selections, prompted_correct_placements) = torch.stack([
            planned_prompts, actual_prompts, conflicts, targets_already_used,
            model_correct_selections, prompted_correct_placements,
        ]).cpu().tolist()
        self.lr_scheduler.step()

        avg_loss = total_loss / float(max(1, total_samples))

        return {
            "loss": avg_loss,
            "teacher_forcing_prob": p_teacher,
            "planned_prompt_ratio": planned_prompts / float(max(1, total_steps)),
            "actual_prompt_ratio": actual_prompts / float(max(1, total_steps)),
            "conflict_ratio": conflicts / float(max(1, total_steps)),
            "target_already_used_ratio": targets_already_used / float(max(1, total_steps)),
            "prompt_failure_ratio": conflicts / float(max(1, planned_prompts)),
            "train_pre_override_acc": model_correct_selections / float(max(1, total_steps)),
            "train_prompted_acc": prompted_correct_placements / float(max(1, total_steps)),
            "lr": epoch_lr,
        }

    @torch.no_grad()
    def validate(self, strategy: Optional[str] = None) -> Dict[str, float]:
        """
        Validate purely autonomously (teacher forcing OFF, p=0).
        """
        self.model.eval()
        eval_strategy = strategy if strategy is not None else self.train_config.expansion_strategy

        total_non_seed_patches = 0
        correct_non_seed_patches = 0
        perfect_puzzles = 0
        total_puzzles = 0
        total_duplicate_cases = 0
        pairwise_sum = 0.0

        for batch in self.val_loader:
            candidates = batch["candidates"]                 # (B, K, 3, P, P)
            seed_cands = batch["seed_cand"]                 # (B,)
            seed_coords = batch["seed_coord"]               # (B, 2)
            target_mappings = batch["target_mapping"]       # (B, G, G)

            B = candidates.shape[0]

            # Solve batch autonomously
            pred_grid, _ = solve_batch(
                model=self.model,
                patches=candidates,
                seed_cands=seed_cands,
                seed_coords=seed_coords,
                grid_size=self.grid_size,
                strategy=eval_strategy,
                rng_seeds=batch.get("expansion_seeds", [104729 + int(i) * 7919 for i in batch["img_indices"]]),
                record_trajectories=False,
            )  # pred_grid is (B, G, G)

            # Verify duplicate rate
            for b in range(B):
                unique_cands = torch.unique(pred_grid[b])
                if len(unique_cands) < self.K:
                    total_duplicate_cases += 1

            # Accuracy metrics
            for b in range(B):
                sr, sc = int(seed_coords[b, 0].item()), int(seed_coords[b, 1].item())
                # Mask out seed position from patch accuracy
                mask = torch.ones((self.grid_size, self.grid_size), dtype=torch.bool)
                mask[sr, sc] = False

                pred_non_seed = pred_grid[b][mask]
                true_non_seed = target_mappings[b][mask]

                correct_matches = (pred_non_seed == true_non_seed)
                n_correct = int(correct_matches.sum().item())
                n_total = len(pred_non_seed)

                correct_non_seed_patches += n_correct
                total_non_seed_patches += n_total

                if n_correct == n_total:
                    perfect_puzzles += 1
                total_puzzles += 1

            # Pairwise accuracy
            pairwise_acc = compute_pairwise_accuracy(pred_grid, target_mappings, self.grid_size)
            pairwise_sum += pairwise_acc * B

        patch_acc = correct_non_seed_patches / float(max(1, total_non_seed_patches))
        perfect_acc = perfect_puzzles / float(max(1, total_puzzles))
        mean_pairwise_acc = pairwise_sum / max(1, total_puzzles)
        duplicate_rate = total_duplicate_cases / float(max(1, total_puzzles))
        if total_puzzles == 0:
            raise ValueError("Validation loader is empty")
        if duplicate_rate:
            raise RuntimeError("Autonomous solver reused a candidate")

        return {
            "val_patch_acc": patch_acc,
            "val_perfect_acc": perfect_acc,
            "val_pairwise_acc": mean_pairwise_acc,
            "val_duplicate_rate": duplicate_rate,
        }

    def save_checkpoint(self, epoch: int, is_best: bool):
        checkpoint = {
            "format_version": 2,
            "validation_protocol": "heldout_train_fixed_expansion_v2",
            "epoch": epoch,
            "grid_size": self.grid_size,
            "model_config": asdict(self.model_config),
            "train_config": asdict(self.train_config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "curriculum_state": self.curriculum.state_dict(),
            "best_acc": self.best_acc,
            "best_perfect_acc": self.best_perfect_acc,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            }
        }

        latest_path = os.path.join(self.save_dir, "latest.pt")
        save_atomic(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best.pt")
            save_atomic(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_model = checkpoint["model_config"]
        if not isinstance(saved_model, dict):
            saved_model = asdict(saved_model)
        if saved_model != asdict(self.model_config):
            raise ValueError("Resume model configuration does not match checkpoint")
        saved_train = checkpoint["train_config"]
        if not isinstance(saved_train, dict):
            saved_train = asdict(saved_train)
        for key in ("epochs", "warmup_epochs", "lr", "weight_decay", "seed", "seed_selection", "expansion_strategy", "batch_size", "seed_coord"):
            if saved_train.get(key) != asdict(self.train_config).get(key):
                raise ValueError(f"Resume configuration differs for {key}; use checkpoint settings")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.curriculum.load_state_dict(checkpoint["curriculum_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint.get("best_acc", -1.0)
        self.best_perfect_acc = checkpoint.get("best_perfect_acc", -1.0)
        if checkpoint.get("validation_protocol") != "heldout_train_fixed_expansion_v2":
            self.best_acc = self.best_perfect_acc = -1.0
            warnings.warn(
                "Legacy checkpoint used the official test set for model selection and trained on "
                "the full training set. Its scores are not comparable to the new held-out protocol; "
                "start a fresh run for an uncontaminated validation split.",
                UserWarning,
            )

        # Restore RNG state if available
        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if rng.get("torch_cuda_all") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda_all"]])
            elif rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng["torch_cuda"].cpu())
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("random") is not None:
                random.setstate(rng["random"])
