"""
Curriculum and Learning Rate Scheduling for v2:
- 3-stage teacher forcing curriculum: 20% / 60% / 20%
- Linear warmup + cosine decay learning rate scheduler
- State serialization for checkpoint recovery
"""

import math
from typing import Dict, Any, Tuple
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CurriculumScheduler:
    """
    Controls teacher-forcing probability across curriculum stages:
    Stage 1 (0% - 20% epochs): p = 1.0 (Full correct context)
    Stage 2 (20% - 80% epochs): p linearly decays from 1.0 to 0.0 (Gradual removal)
    Stage 3 (80% - 100% epochs): p = 0.0 (Autonomous assembly)
    """
    def __init__(self, total_epochs: int = 100):
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        self.total_epochs = total_epochs
        self.stage1_end = 1 if total_epochs < 3 else min(total_epochs - 2, max(1, round(0.2 * total_epochs)))
        self.stage2_end = (self.stage1_end if total_epochs < 3 else
                           min(total_epochs - 1, max(self.stage1_end + 1, round(0.8 * total_epochs))))

    def get_stage_info(self, epoch: int) -> Dict[str, Any]:
        """
        epoch is 1-indexed (1 to total_epochs).
        """
        if epoch <= self.stage1_end:
            stage_name = "correct_context"
            p = 1.0
        elif epoch <= self.stage2_end:
            stage_name = "gradual_removal"
            progress = (epoch - self.stage1_end) / max(1, (self.stage2_end - self.stage1_end))
            p = max(0.0, min(1.0, 1.0 - progress))
        else:
            stage_name = "autonomous"
            p = 0.0

        return {
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "stage_name": stage_name,
            "stage1_end": self.stage1_end,
            "stage2_end": self.stage2_end,
            "teacher_forcing_prob": float(p),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_epochs": self.total_epochs,
            "stage1_end": self.stage1_end,
            "stage2_end": self.stage2_end,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.total_epochs = state["total_epochs"]
        self.stage1_end = state["stage1_end"]
        self.stage2_end = state["stage2_end"]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-5
) -> _LRScheduler:
    """
    Warmup + Cosine Annealing LR Scheduler.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_epochs = min(warmup_epochs, max(0, total_epochs - 1))
    min_ratio = min(1.0, min_lr / float(max(1e-8, base_lr)))

    def lr_lambda(epoch: int) -> float:
        # epoch is 0-indexed in PyTorch scheduler
        current_epoch = epoch + 1
        if current_epoch <= warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            progress = min(1.0, max(0.0, (current_epoch - warmup_epochs - 1) / float(max(1, total_epochs - warmup_epochs - 1))))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
