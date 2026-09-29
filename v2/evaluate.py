"""
Evaluation module for v2:
Evaluates a checkpoint using 3 pre-fixed random configurations for both BFS and Random Frontier.
Reports non-seed patch accuracy, whole puzzle perfect accuracy, pairwise relationship accuracy,
and mean ± std across runs. Asserts duplicate rate is strictly 0.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, GridConfig
from .model import JigsawSolverV2
from .dataset import CIFAR10JigsawDataset, collate_jigsaw
from .solver import solve_batch
from .trainer import compute_pairwise_accuracy


def evaluate_single_config(
    model: JigsawSolverV2,
    val_loader: DataLoader,
    grid_size: int,
    strategy: str,
    device: torch.device
) -> Dict[str, float]:
    """
    Run evaluation on the dataset with a specific expansion strategy.
    """
    model.eval()
    K = grid_size * grid_size

    total_non_seed_patches = 0
    correct_non_seed_patches = 0
    perfect_puzzles = 0
    total_puzzles = 0
    duplicate_cases = 0
    pairwise_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            candidates = batch["candidates"].to(device)
            seed_cands = batch["seed_cand"].to(device)
            seed_coords = batch["seed_coord"].to(device)
            target_mappings = batch["target_mapping"].to(device)
            B = candidates.shape[0]

            pred_grid, _ = solve_batch(
                model=model,
                patches=candidates,
                seed_cands=seed_cands,
                seed_coords=seed_coords,
                grid_size=grid_size,
                strategy=strategy,
                rng_seeds=batch.get("expansion_seeds", [104729 + int(i) * 7919 for i in batch["img_indices"]]),
                record_trajectories=False,
            )
            pred_grid = pred_grid.to(device)

            # Check duplicate rate
            for b in range(B):
                if len(torch.unique(pred_grid[b])) < K:
                    duplicate_cases += 1

                sr, sc = int(seed_coords[b, 0].item()), int(seed_coords[b, 1].item())
                mask = torch.ones((grid_size, grid_size), dtype=torch.bool, device=device)
                mask[sr, sc] = False

                pred_non_seed = pred_grid[b][mask]
                true_non_seed = target_mappings[b][mask]

                correct = (pred_non_seed == true_non_seed)
                n_corr = int(correct.sum().item())
                n_tot = len(pred_non_seed)

                correct_non_seed_patches += n_corr
                total_non_seed_patches += n_tot
                if n_corr == n_tot:
                    perfect_puzzles += 1
                total_puzzles += 1

            pairwise_acc = compute_pairwise_accuracy(pred_grid, target_mappings, grid_size)
            pairwise_sum += pairwise_acc * B

    patch_acc = correct_non_seed_patches / float(max(1, total_non_seed_patches))
    perfect_acc = perfect_puzzles / float(max(1, total_puzzles))
    pairwise_acc_mean = pairwise_sum / max(1, total_puzzles)
    if total_puzzles == 0:
        raise ValueError("Evaluation loader is empty")
    duplicate_rate = duplicate_cases / float(max(1, total_puzzles))

    assert duplicate_rate == 0.0, f"Duplicate rate must be 0, got {duplicate_rate}"

    return {
        "patch_acc": patch_acc,
        "perfect_acc": perfect_acc,
        "pairwise_acc": pairwise_acc_mean,
        "duplicate_rate": duplicate_rate,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str = "/home/cjc/桌面/myidea/data/cifar10",
    batch_size: int = 64,
    device_str: str = "cuda",
    eval_seeds: Optional[List[int]] = None,
    max_eval_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Loads checkpoint, builds model, and runs evaluation on BFS and Random Frontier
    across 3 fixed seeds.
    """
    if eval_seeds is None:
        eval_seeds = [42, 101, 202]
    if not eval_seeds:
        raise ValueError("At least one evaluation seed is required")
    if max_eval_samples is not None and max_eval_samples <= 0:
        raise ValueError("max_eval_samples must be positive")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config: ModelConfig = checkpoint["model_config"]
    if isinstance(model_config, dict):
        model_config = ModelConfig(**model_config)
    grid_size = checkpoint["grid_size"]

    model = JigsawSolverV2(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    strategies = ["bfs", "random_frontier"]
    results: Dict[str, Any] = {}

    for strat in strategies:
        strat_results: Dict[str, List[float]] = {
            "patch_acc": [],
            "perfect_acc": [],
            "pairwise_acc": [],
            "duplicate_rate": [],
        }

        for seed in eval_seeds:
            val_dataset = CIFAR10JigsawDataset(
                root=data_dir,
                grid_size=grid_size,
                train=False,
                seed_mode="random",
                base_seed=seed,
                download=False
            )

            if max_eval_samples is not None and max_eval_samples < len(val_dataset):
                val_dataset = torch.utils.data.Subset(val_dataset, list(range(max_eval_samples)))

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_jigsaw
            )

            metrics = evaluate_single_config(
                model=model,
                val_loader=val_loader,
                grid_size=grid_size,
                strategy=strat,
                device=device
            )

            for k, v in metrics.items():
                strat_results[k].append(v)

        summary = {}
        for k in ["patch_acc", "perfect_acc", "pairwise_acc", "duplicate_rate"]:
            vals = strat_results[k]
            summary[f"{k}_runs"] = vals
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

        results[strat] = summary

    return results
