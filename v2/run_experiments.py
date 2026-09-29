"""
CLI runner for training, overfitting checks, ablation studies, and evaluation in v2.
"""

import argparse
import json
import os
import random
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import ModelConfig, TrainConfig
from .model import JigsawSolverV2
from .dataset import CIFAR10JigsawDataset, collate_jigsaw, build_train_val_datasets
from .trainer import TrainerV2
from .evaluate import evaluate_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    grid_size: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    mode: str = "both",
    expansion_strategy: str = "random_frontier",
    seed_selection: str = "random",
    lr: float = 3e-4,
    data_dir: str = "/home/cjc/桌面/myidea/data/cifar10",
    save_dir_prefix: str = "checkpoints/v2",
    resume_from: str = None,
    max_train_samples: int = None,
    max_val_samples: int = None,
    seed: int = 42,
    seed_coord=None,
    num_workers: int = 4,
    device_str: str = None,
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_config = ModelConfig(
        grid_size=grid_size,
        content_dim=96,
        local_heads=4,
        global_layers=2,
        global_heads=4,
        mode=mode
    )

    if mode != "both" or expansion_strategy != "random_frontier":
        save_dir_prefix = os.path.join(save_dir_prefix, mode, expansion_strategy)
    train_config = TrainConfig(
        data_dir=data_dir,
        save_dir_prefix=save_dir_prefix,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        expansion_strategy=expansion_strategy,
        seed_selection=seed_selection,
        device=str(device),
        seed=seed,
        seed_coord=tuple(seed_coord) if seed_coord is not None else None,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    if resume_from is not None:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(resume_from)
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        mc, tc = saved["model_config"], saved["train_config"]
        model_config = ModelConfig(**(mc if isinstance(mc, dict) else asdict(mc)))
        train_config = TrainConfig(**(tc if isinstance(tc, dict) else asdict(tc)))
        train_config.device = str(device)
        print("Restoring saved model, data split, curriculum and training configuration.")
    print(f"=== Training v2: grid={model_config.grid_size}, mode={model_config.mode}, "
          f"strategy={train_config.expansion_strategy}, device={device} ===")
    set_seed(train_config.seed)
    train_dataset, val_dataset = build_train_val_datasets(
        root=train_config.data_dir,
        grid_size=model_config.grid_size,
        seed_mode=train_config.seed_selection,
        seed_coord=train_config.seed_coord,
        base_seed=train_config.seed,
        max_train_samples=train_config.max_train_samples,
        max_val_samples=train_config.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_jigsaw,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_jigsaw,
        pin_memory=torch.cuda.is_available()
    )

    model = JigsawSolverV2(model_config)
    trainer = TrainerV2(
        model=model,
        model_config=model_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

    for epoch in range(trainer.start_epoch, train_config.epochs + 1):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate()

        val_patch_acc = val_metrics["val_patch_acc"]
        val_perf_acc = val_metrics["val_perfect_acc"]

        is_best = False
        if val_patch_acc > trainer.best_acc:
            is_best = True
            trainer.best_acc = val_patch_acc
            trainer.best_perfect_acc = val_perf_acc
        elif abs(val_patch_acc - trainer.best_acc) < 1e-6 and val_perf_acc > trainer.best_perfect_acc:
            is_best = True
            trainer.best_perfect_acc = val_perf_acc

        trainer.save_checkpoint(epoch, is_best=is_best)

        print(
            f"Epoch {epoch:03d}/{train_config.epochs:03d} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"TF Prob: {train_metrics['teacher_forcing_prob']:.2f} | "
            f"Plan/Act/Conf: {train_metrics['planned_prompt_ratio']:.2f}/{train_metrics['actual_prompt_ratio']:.2f}/{train_metrics['conflict_ratio']:.2f} | "
            f"UsedGT: {train_metrics['target_already_used_ratio']:.2f} | "
            f"Tr Context/Placed Acc: {train_metrics['train_pre_override_acc']*100:.1f}%/{train_metrics['train_prompted_acc']*100:.1f}% | "
            f"Val Patch/Perf: {val_patch_acc*100:.1f}%/{val_perf_acc*100:.1f}% | "
            f"Best: {trainer.best_acc*100:.1f}% {'*' if is_best else ''}"
        )

    print("Training complete!")
    return trainer.save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2 Jigsaw Runner")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "local_only", "global_only"])
    parser.add_argument("--grid-size", type=int, default=3, choices=[3, 5, 7])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--strategy", type=str, default="random_frontier", choices=["random_frontier", "bfs"])
    parser.add_argument("--seed-selection", type=str, default="random", choices=["random", "center", "custom"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    parser.add_argument("--data-dir", default="/home/cjc/桌面/myidea/data/cifar10")
    parser.add_argument("--save-dir-prefix", default="checkpoints/v2")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-coord", type=int, nargs=2, metavar=("ROW", "COL"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)

    args = parser.parse_args()

    if args.eval_checkpoint is not None:
        print(f"Evaluating checkpoint: {args.eval_checkpoint}")
        results = evaluate_checkpoint(args.eval_checkpoint, data_dir=args.data_dir, batch_size=args.batch_size,
                                      device_str=args.device or "cuda", max_eval_samples=args.max_eval_samples)
        print(json.dumps(results, indent=2))
    else:
        run_training(
            grid_size=args.grid_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mode=args.mode,
            expansion_strategy=args.strategy,
            seed_selection=args.seed_selection,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            data_dir=args.data_dir,
            save_dir_prefix=args.save_dir_prefix,
            resume_from=args.resume,
            seed=args.seed,
            seed_coord=args.seed_coord,
            num_workers=args.num_workers,
            device_str=args.device,
        )
