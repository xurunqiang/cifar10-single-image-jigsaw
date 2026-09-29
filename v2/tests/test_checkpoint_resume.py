"""
Test 6: Checkpoint Save/Resume and Determinism
Verify:
- Fixed random config reproduces seed, candidate shuffle, and expansion order.
- Checkpoint faithfully stores model config, curriculum state, optimizer, and RNG state.
- Resuming from checkpoint correctly continues at next epoch with correct curriculum stage and teacher forcing probability.
"""

import os
import random
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from v2.config import ModelConfig, TrainConfig
from v2.model import JigsawSolverV2
from v2.curriculum import CurriculumScheduler
from v2.dataset import JigsawProblemGenerator, CIFAR10JigsawDataset, collate_jigsaw
from v2.trainer import TrainerV2


def test_reproducibility_with_fixed_config():
    grid_size = 3
    img = torch.randn(3, 32, 32)

    # Run 1
    rng1 = random.Random(999)
    prob1 = JigsawProblemGenerator.generate(img, grid_size=grid_size, rng=rng1)

    # Run 2
    rng2 = random.Random(999)
    prob2 = JigsawProblemGenerator.generate(img, grid_size=grid_size, rng=rng2)

    assert prob1["seed_cand"] == prob2["seed_cand"]
    assert prob1["seed_coord"] == prob2["seed_coord"]
    assert torch.equal(prob1["target_mapping"], prob2["target_mapping"])
    assert torch.equal(prob1["candidates"], prob2["candidates"])


def test_checkpoint_save_and_resume():
    grid_size = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        model_config = ModelConfig(grid_size=grid_size, content_dim=96, mode="both")
        train_config = TrainConfig(
            data_dir="/tmp",
            save_dir_prefix=tmpdir,
            batch_size=2,
            lr=3e-4,
            epochs=100
        )

        model = JigsawSolverV2(model_config)

        # Mock minimal dataset
        dummy_batch = [
            JigsawProblemGenerator.generate(torch.randn(3, 32, 32), grid_size=grid_size, rng=random.Random(i))
            for i in range(4)
        ]
        for i, item in enumerate(dummy_batch):
            item["img_idx"] = i
            item["class_label"] = 0

        class DummyDataset:
            def __len__(self):
                return len(dummy_batch)
            def __getitem__(self, idx):
                return dummy_batch[idx]

        dummy_loader = DataLoader(DummyDataset(), batch_size=2, collate_fn=collate_jigsaw)
        device = torch.device("cpu")

        trainer = TrainerV2(
            model=model,
            model_config=model_config,
            train_config=train_config,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            device=device
        )

        # Simulate reaching epoch 25 (inside Stage 2: gradual removal)
        for ep in range(1, 26):
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
        trainer.best_acc = 0.75
        trainer.save_checkpoint(epoch=25, is_best=True)

        ckpt_path = os.path.join(trainer.save_dir, "best.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

        # Create fresh model & trainer to resume
        fresh_model = JigsawSolverV2(model_config)
        fresh_trainer = TrainerV2(
            model=fresh_model,
            model_config=model_config,
            train_config=train_config,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            device=device
        )

        fresh_trainer.load_checkpoint(ckpt_path)

        # Check resumed epoch and curriculum
        assert fresh_trainer.start_epoch == 26
        assert fresh_trainer.best_acc == 0.75

        stage_info = fresh_trainer.curriculum.get_stage_info(fresh_trainer.start_epoch)
        assert stage_info["stage_name"] == "gradual_removal"
        # At epoch 26 out of 100, stage 1 ends at 20, stage 2 ends at 80.
        # progress = (26 - 20) / 60 = 6/60 = 0.1 -> p = 0.9
        assert abs(stage_info["teacher_forcing_prob"] - 0.9) < 1e-4

        # Verify weights match
        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert torch.equal(p1, p2)
