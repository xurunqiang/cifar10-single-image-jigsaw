"""
Joint Trainer for v3:
- Dual-view training with Jigsaw Puzzle + VICReg Self-Supervised Loss
- Disentangled masking (classification on all non-seed candidates; action on unused candidates)
- Autonomous board inference for semantic branch during Teacher Forcing
- BF16 mixed-precision training with FP32 VICReg statistics
- Gradient clipping (1.0) and atomic checkpointing (latest.pt, best_jigsaw.pt)
"""

import os
import random
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .model import JigsawSolverV3
from .vicreg import VICRegProjector, VICRegLoss
from .curriculum import CurriculumScheduler, build_lr_scheduler
from .solver import solve_batch, compute_puzzle_accuracy, compute_neighbor_accuracy
from .dataset import generate_bfs_order


class JigsawTrainerV3:
    """
    Unified Trainer for v3: Jigsaw Puzzle + VICReg Whole-Image Representation Learning.
    """
    def __init__(
        self,
        model: JigsawSolverV3,
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        run_metadata: Optional[Dict[str, Any]] = None
    ):
        self.model = model.to(device)
        self.model_config = model_config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_metadata = run_metadata or {}
        self.grid_size = model_config.grid_size
        self.K = self.grid_size * self.grid_size
        self.bfs_order = generate_bfs_order(self.grid_size, (self.grid_size // 2, self.grid_size // 2))

        # VICReg Projector
        self.projector = VICRegProjector(
            in_dim=model_config.content_dim,
            hidden_dim=model_config.proj_hidden_dim,
            out_dim=model_config.proj_out_dim
        ).to(device)

        # Joint Optimizer
        all_params = list(self.model.parameters()) + list(self.projector.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay
        )

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_epochs=train_config.warmup_epochs,
            total_epochs=train_config.epochs,
            min_lr=train_config.min_lr
        )

        self.curriculum = CurriculumScheduler(
            total_epochs=train_config.epochs,
            semantic_warmup_epochs=train_config.semantic_warmup_epochs,
            semantic_weight_start=train_config.semantic_weight_start,
            semantic_weight_max=train_config.semantic_weight_max
        )

        self.jigsaw_criterion = nn.CrossEntropyLoss()
        self.vicreg_criterion = VICRegLoss(
            sim_weight=train_config.sim_weight,
            var_weight=train_config.var_weight,
            cov_weight=train_config.cov_weight,
            var_threshold=train_config.var_threshold,
            var_eps=train_config.var_eps
        )

        self.start_epoch = 1
        self.best_acc = -1.0
        self.best_perfect_acc = -1.0
        self.current_epoch = 0
        self.best_feat_acc = -1.0
        self.history = []

        self.save_dir = train_config.get_save_dir()
        os.makedirs(self.save_dir, exist_ok=True)

    def _process_jigsaw_view(
        self,
        view_data: Dict[str, Any],
        p_teacher: float,
        use_amp: bool,
        compute_semantic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes jigsaw solving for a single augmented view:
        - Computes cross-entropy loss over 24 BFS steps
        - Produces virtual patch Z_virtual on autonomous board
        """
        candidates = view_data["candidates"].to(self.device)         # (B, K, 3, 13, 13)
        seed_cands = view_data["seed_cand"].to(self.device)           # (B,)
        seed_coords = view_data["seed_coord"].to(self.device)         # (B, 2)
        target_mappings = view_data["target_mapping"].to(self.device) # (B, G, G)

        B = candidates.shape[0]
        batch_idx = torch.arange(B, device=self.device)

        # 1. Encode candidate patches once
        raw_feats = self.model.encode_candidates(candidates)          # (B, K, D)

        # 2. Sequential jigsaw curriculum loop
        grid_placed = torch.full((B, self.grid_size, self.grid_size), -1, dtype=torch.long, device=self.device)
        used_candidates = torch.zeros((B, self.K), dtype=torch.bool, device=self.device)
        cand_to_slot = torch.full((B, self.K, 2), -1, dtype=torch.long, device=self.device)

        # Place center seed (2, 2)
        grid_placed[batch_idx, seed_coords[:, 0], seed_coords[:, 1]] = seed_cands
        used_candidates[batch_idx, seed_cands] = True
        cand_to_slot[batch_idx, seed_cands] = seed_coords

        batch_jigsaw_loss = torch.tensor(0.0, device=self.device)
        model_correct_count = torch.zeros((), dtype=torch.long, device=self.device)
        planned = torch.zeros_like(model_correct_count)
        actual = torch.zeros_like(model_correct_count)
        conflicts = torch.zeros_like(model_correct_count)
        used_gt = torch.zeros_like(model_correct_count)

        for r, c in self.bfs_order:
            target_coords = torch.tensor([[r, c]], dtype=torch.long, device=self.device).expand(B, 2)
            true_cands = target_mappings[batch_idx, r, c]  # (B,)

            logits, _ = self.model.score_step(
                raw_feats=raw_feats,
                grid_placed=grid_placed,
                used_candidates=used_candidates,
                cand_to_slot=cand_to_slot,
                target_coords=target_coords,
                grid_size=self.grid_size
            )  # (B, K)

            # Disentangled Classification Loss: evaluate on all non-seed candidates
            loss_logits = logits.clone()
            loss_logits[batch_idx, seed_cands] = -float("inf")
            step_loss = self.jigsaw_criterion(loss_logits, true_cands)
            batch_jigsaw_loss = batch_jigsaw_loss + step_loss

            # Disentangled Action Selection: evaluate only on unused candidates
            selection_logits = logits.masked_fill(used_candidates, -float("inf"))
            pred_cands = torch.argmax(selection_logits, dim=-1)

            model_correct = (pred_cands == true_cands)
            model_correct_count += model_correct.sum()

            # Teacher Forcing execution
            u = torch.rand(B, device=self.device)
            prompt_draw = (u < p_teacher)
            tc_used = used_candidates[batch_idx, true_cands]
            can_prompt = prompt_draw & (~tc_used)
            planned += prompt_draw.sum()
            actual += can_prompt.sum()
            conflicts += (prompt_draw & tc_used).sum()
            used_gt += tc_used.sum()
            actual_placed = torch.where(can_prompt, true_cands, pred_cands)

            # Out-of-place state update
            grid_placed = grid_placed.clone()
            grid_placed[batch_idx, r, c] = actual_placed

            used_candidates = used_candidates.clone()
            used_candidates[batch_idx, actual_placed] = True

            cand_to_slot = cand_to_slot.clone()
            cand_to_slot[batch_idx, actual_placed] = target_coords

        num_steps = len(self.bfs_order)  # 24
        avg_jigsaw_loss = batch_jigsaw_loss / float(num_steps)

        # 3. Autonomous board placement for semantic branch
        if not compute_semantic:
            z_virtual = None
        elif p_teacher > 0.0:
            # Run autonomous solver in no_grad to get model's self-predicted board
            with torch.no_grad():
                auto_out = solve_batch(
                    model=self.model,
                    patches=candidates,
                    seed_cands=seed_cands,
                    seed_coords=seed_coords,
                    grid_size=self.grid_size,
                    expansion_order=self.bfs_order,
                    raw_feats=raw_feats.detach(),
                    compute_representation=False
                )
                auto_cand_to_slot = auto_out["cand_to_slot"]
                auto_used = torch.ones((B, self.K), dtype=torch.bool, device=self.device)

            # Compute virtual patch with autograd through raw_feats and transformer
            z_virtual, _ = self.model.compute_virtual_patch(raw_feats, auto_cand_to_slot, auto_used)
        else:
            # Autonomous phase: the sequential loop was already fully autonomous!
            z_virtual, _ = self.model.compute_virtual_patch(raw_feats, cand_to_slot, used_candidates)

        stats = {
            "model_correct": model_correct_count,
            "planned": planned, "actual": actual, "conflicts": conflicts, "used_gt": used_gt,
            "total_slots": B * num_steps
        }

        return avg_jigsaw_loss, z_virtual, stats

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.current_epoch = int(epoch)
        self.model.train()
        self.projector.train()

        stage_info = self.curriculum.get_stage_info(epoch)
        p_teacher = stage_info["tf_prob"]
        lambda_sem = stage_info["semantic_weight"]

        total_loss = 0.0
        total_jigsaw_loss = 0.0
        total_vicreg_loss = 0.0
        total_samples = 0
        total_model_correct = torch.zeros((), dtype=torch.long, device=self.device)
        total_slots = 0
        prompt_totals = {k: torch.zeros((), device=self.device) for k in ("planned", "actual", "conflicts", "used_gt")}
        vic_totals = {k: 0.0 for k in ("sim_loss", "var_loss", "cov_loss", "std_mean")}
        epoch_lr = self.optimizer.param_groups[0]["lr"]

        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.train_config.epochs} [Train]", dynamic_ncols=True)
        for batch in pbar:
            B = batch["view1"]["candidates"].shape[0]
            total_samples += B
            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Process view 1
                loss_j1, z_virt_1, stats1 = self._process_jigsaw_view(batch["view1"], p_teacher, use_amp, lambda_sem > 0)
                # Process view 2
                loss_j2, z_virt_2, stats2 = self._process_jigsaw_view(batch["view2"], p_teacher, use_amp, lambda_sem > 0)

                # Combined jigsaw loss
                loss_jigsaw = 0.5 * (loss_j1 + loss_j2)

                if lambda_sem > 0:
                    y1, y2 = self.projector(z_virt_1), self.projector(z_virt_2)
                    loss_vicreg, v_stats = self.vicreg_criterion(y1, y2)
                else:
                    loss_vicreg = loss_jigsaw.new_zeros(())
                    v_stats = {k: 0.0 for k in vic_totals}

                # Total joint loss
                loss_total = loss_jigsaw + lambda_sem * loss_vicreg

            loss_total.backward()

            # Gradient clipping across model and projector
            all_params = list(self.model.parameters()) + list(self.projector.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.train_config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()

            total_loss += float(loss_total.item()) * B
            total_jigsaw_loss += float(loss_jigsaw.item()) * B
            total_vicreg_loss += float(loss_vicreg.item()) * B
            total_model_correct += stats1["model_correct"] + stats2["model_correct"]
            total_slots += stats1["total_slots"] + stats2["total_slots"]
            for key in prompt_totals:
                prompt_totals[key] += stats1[key] + stats2[key]
            for key in vic_totals:
                vic_totals[key] += v_stats[key] * B

            pbar.set_postfix({
                "loss": f"{loss_total.item():.4f}",
                "jig": f"{loss_jigsaw.item():.4f}",
                "vic": f"{loss_vicreg.item():.4f}",
                "tf": f"{p_teacher:.2f}"
            })

        if total_samples == 0:
            raise ValueError("Training loader has no complete batch")
        self.lr_scheduler.step()

        avg_loss = total_loss / max(1, total_samples)
        avg_j_loss = total_jigsaw_loss / max(1, total_samples)
        avg_v_loss = total_vicreg_loss / max(1, total_samples)
        train_patch_acc = (float(total_model_correct.item()) / max(1, total_slots)) * 100.0

        return {
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_jigsaw": avg_j_loss,
            "loss_vicreg": avg_v_loss,
            "train_patch_acc": train_patch_acc,
            "tf_prob": p_teacher,
            "semantic_weight": lambda_sem,
            "lr": epoch_lr,
            **{k + "_ratio": float(v.item()) / max(1, total_slots) for k, v in prompt_totals.items()},
            **{k: v / total_samples for k, v in vic_totals.items()}
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Runs validation over val_loader using autonomous solver.
        Evaluates non-seed patch accuracy and perfect assembly rate.
        """
        self.model.eval()
        self.projector.eval()

        total_samples = 0
        total_patch_acc = 0.0
        total_perfect = 0.0
        total_neighbor = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch:03d} [Val]", leave=False, dynamic_ncols=True)
        for batch in pbar:
            candidates = batch["candidates"].to(self.device)
            seed_cands = batch["seed_cand"].to(self.device)
            seed_coords = batch["seed_coord"].to(self.device)
            target_mappings = batch["target_mapping"].to(self.device)
            B = candidates.shape[0]

            out = solve_batch(
                model=self.model,
                patches=candidates,
                seed_cands=seed_cands,
                seed_coords=seed_coords,
                grid_size=self.grid_size,
                expansion_order=self.bfs_order
            )

            patch_acc, perfect = compute_puzzle_accuracy(
                grid_placed=out["grid_placed"],
                target_mapping=target_mappings,
                seed_coords=seed_coords,
                grid_size=self.grid_size
            )

            total_samples += B
            total_patch_acc += float(patch_acc.sum().item())
            total_perfect += float(perfect.sum().item())
            total_neighbor += float(compute_neighbor_accuracy(out["grid_placed"], target_mappings).sum().item())

        val_patch_acc = (total_patch_acc / max(1, total_samples)) * 100.0
        val_perfect_acc = (total_perfect / max(1, total_samples)) * 100.0

        return {
            "val_patch_acc": val_patch_acc,
            "val_perfect_acc": val_perfect_acc,
            "val_neighbor_acc": 100 * total_neighbor / max(1, total_samples),
            "total_val_samples": total_samples
        }

    def save_checkpoint(self, filename: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Atomically saves training checkpoint to filename in self.save_dir.
        """
        filepath = os.path.join(self.save_dir, filename)
        tmp_filepath = filepath + ".tmp"

        payload = {
            "format_version": 2,
            "epoch": self.current_epoch,
            "run_metadata": self.run_metadata,
            "best_feat_acc": self.best_feat_acc,
            "history": self.history,
            "model_state_dict": self.model.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "curriculum_state_dict": self.curriculum.state_dict(),
            "model_config": asdict(self.model_config),
            "train_config": asdict(self.train_config),
            "best_acc": self.best_acc,
            "best_perfect_acc": self.best_perfect_acc,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "train_loader_generator": (
                    self.train_loader.generator.get_state()
                    if getattr(self.train_loader, "generator", None) is not None
                    else None
                ),
            },
        }
        if extra_meta:
            payload.update(extra_meta)

        torch.save(payload, tmp_filepath)
        os.replace(tmp_filepath, filepath)
        return filepath

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Loads training checkpoint from filepath.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if checkpoint.get("format_version") != 2:
            raise ValueError("This checkpoint predates the planned virtual-query/projector architecture; start a new v3 run")
        canonical = lambda value: json.dumps(value, sort_keys=True)
        if canonical(checkpoint["model_config"]) != canonical(asdict(self.model_config)):
            raise ValueError("Resume model configuration differs from checkpoint")
        saved_train = dict(checkpoint["train_config"])
        current_train = asdict(self.train_config)
        for runtime_key in ("device", "save_dir"):
            saved_train.pop(runtime_key, None)
            current_train.pop(runtime_key, None)
        if canonical(saved_train) != canonical(current_train):
            raise ValueError("Resume training configuration differs from checkpoint; keep the saved schedule/batch/seed")
        if canonical(checkpoint.get("run_metadata", {})) != canonical(self.run_metadata):
            raise ValueError("Resume dataset split or augmentation configuration differs from checkpoint")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.projector.load_state_dict(checkpoint["projector_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.curriculum.load_state_dict(checkpoint["curriculum_state_dict"])

        self.best_acc = checkpoint.get("best_acc", -1.0)
        self.best_perfect_acc = checkpoint.get("best_perfect_acc", -1.0)
        self.current_epoch = int(checkpoint["epoch"])
        self.start_epoch = self.current_epoch + 1
        self.best_feat_acc = checkpoint.get("best_feat_acc", -1.0)
        self.history = checkpoint.get("history", [])

        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            saved_cuda_states = rng_state.get("cuda", [])
            if torch.cuda.is_available() and len(saved_cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_states])
            loader_generator = getattr(self.train_loader, "generator", None)
            if loader_generator is not None and rng_state.get("train_loader_generator") is not None:
                loader_generator.set_state(rng_state["train_loader_generator"].cpu())

        print(f"Loaded checkpoint from {filepath} (resuming at epoch {self.start_epoch})")
        return checkpoint
