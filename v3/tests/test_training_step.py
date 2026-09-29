"""
Integration test for full joint training step, validation, and checkpointing.
"""

import tempfile
import shutil
import torch
from torch.utils.data import DataLoader, Dataset

from v3.config import ModelConfig, TrainConfig
from v3.model import JigsawSolverV3
from v3.trainer import JigsawTrainerV3
from v3.dataset import collate_two_views, collate_single_view, make_puzzle_dict


class DummyDualDataset(Dataset):
    def __init__(self, size=4):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img1 = torch.randn(3, 64, 64)
        img2 = torch.randn(3, 64, 64)
        p1 = make_puzzle_dict(img1, grid_size=5, seed_coord=(2, 2))
        p2 = make_puzzle_dict(img2, grid_size=5, seed_coord=(2, 2))
        return {
            "view1": p1,
            "view2": p2,
            "class_idx": idx % 10,
            "wnid": f"n{idx:05d}",
            "rel_path": f"dummy_{idx}.JPEG"
        }


class DummySingleDataset(Dataset):
    def __init__(self, size=4):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, 64, 64)
        p = make_puzzle_dict(img, grid_size=5, seed_coord=(2, 2))
        return {
            "candidates": p["candidates"],
            "seed_cand": p["seed_cand"],
            "seed_coord": p["seed_coord"],
            "target_mapping": p["target_mapping"],
            "cand_perm": p["cand_perm"],
            "orig_shape": p["orig_shape"],
            "raw_img": img,
            "class_idx": idx % 10,
            "wnid": f"n{idx:05d}",
            "rel_path": f"dummy_{idx}.JPEG"
        }


def test_full_trainer_step_and_checkpoint():
    device = torch.device("cpu")
    tmp_dir = tempfile.mkdtemp()

    try:
        m_cfg = ModelConfig(
            content_dim=64,
            local_layers=1,
            local_heads=2,
            global_layers=1,
            global_heads=2,
            global_ffn_dim=128,
            proj_hidden_dim=128,
            proj_out_dim=128,
            cnn_stages=(16, 32, 64),
            gradient_checkpointing=False
        )
        t_cfg = TrainConfig(
            batch_size=2,
            epochs=2,
            save_dir=tmp_dir,
            device="cpu",
            num_workers=0
        )

        model = JigsawSolverV3(m_cfg)
        train_generator = torch.Generator().manual_seed(1234)
        train_loader = DataLoader(
            DummyDualDataset(4), batch_size=2, collate_fn=collate_two_views,
            generator=train_generator
        )
        val_loader = DataLoader(DummySingleDataset(4), batch_size=2, collate_fn=collate_single_view)

        trainer = JigsawTrainerV3(
            model=model,
            model_config=m_cfg,
            train_config=t_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )

        # Final epoch of a two-epoch schedule: autonomous stage.
        stats_ep2 = trainer.train_epoch(2)
        assert stats_ep2["loss_total"] > 0
        assert stats_ep2["tf_prob"] == 0.0

        val_stats = trainer.validate()
        assert "val_patch_acc" in val_stats

        # Save checkpoint
        ckpt_path = trainer.save_checkpoint("test_ckpt.pt")

        # Load checkpoint into fresh trainer
        model2 = JigsawSolverV3(m_cfg)
        train_loader2 = DataLoader(
            DummyDualDataset(4), batch_size=2, collate_fn=collate_two_views,
            generator=torch.Generator().manual_seed(999)
        )
        trainer2 = JigsawTrainerV3(
            model=model2,
            model_config=m_cfg,
            train_config=t_cfg,
            train_loader=train_loader2,
            val_loader=val_loader,
            device=device
        )
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.start_epoch == 3
        assert torch.equal(train_loader.generator.get_state(), train_loader2.generator.get_state())

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
