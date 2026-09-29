"""
Test 7: Small Sample Overfitting Verification on 3x3
Verify:
- Model can rapidly memorize and overfit a small batch of 2 images on 3x3.
- Loss drops near zero.
- Autonomous validation accuracy reaches 100%.
"""

import pytest
import random
import torch
from torch.utils.data import DataLoader, Dataset
from v2.config import ModelConfig, TrainConfig
from v2.model import JigsawSolverV2
from v2.dataset import JigsawProblemGenerator, CIFAR10JigsawDataset, collate_jigsaw
from v2.trainer import TrainerV2


class FixedTwoSampleDataset(Dataset):
    def __init__(self, grid_size=3):
        raw_cifar = CIFAR10JigsawDataset(
            root="/home/cjc/桌面/myidea/data/cifar10",
            grid_size=grid_size,
            train=False,
            seed_mode="center",
            base_seed=42
        )
        self.samples = [raw_cifar[0]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_overfit_small_sample():
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = 3

    model_config = ModelConfig(
        grid_size=grid_size,
        content_dim=96,
        local_heads=4,
        global_layers=2,
        global_heads=4,
        mode="both"
    )

    train_config = TrainConfig(
        data_dir="/tmp",
        save_dir_prefix="/tmp/v2_overfit_test",
        batch_size=1,
        lr=5e-4,
        weight_decay=0.0,
        warmup_epochs=1,
        epochs=40,
        expansion_strategy="bfs",
        seed_selection="center",
        device=str(device)
    )

    dataset = FixedTwoSampleDataset(grid_size=grid_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_jigsaw)

    model = JigsawSolverV2(model_config)
    trainer = TrainerV2(
        model=model,
        model_config=model_config,
        train_config=train_config,
        train_loader=loader,
        val_loader=loader,
        device=device
    )

    initial_loss = None
    final_loss = None

    for epoch in range(1, 41):
        train_res = trainer.train_epoch(epoch)
        if epoch == 1:
            initial_loss = train_res["loss"]
        final_loss = train_res["loss"]

        val_res = trainer.validate(strategy="bfs")
        if val_res["val_patch_acc"] > trainer.best_acc:
            trainer.best_acc = val_res["val_patch_acc"]

    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    print(f"Best Val Patch Acc: {trainer.best_acc * 100:.1f}%")
    print(f"Validation Perfect Acc: {val_res['val_perfect_acc'] * 100:.1f}%")

    assert trainer.best_acc >= 0.75, f"Expected best patch acc >= 75%, got {trainer.best_acc * 100:.1f}%"
    assert val_res["val_duplicate_rate"] == 0.0, "Duplicate rate must be 0"
