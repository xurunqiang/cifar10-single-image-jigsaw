"""
Curriculum, Loss Weight, and Learning Rate Scheduling for v3:
- 3-stage Teacher Forcing schedule: 1.0 (ep 1-20) -> linear decay (ep 21-80) -> 0.0 (ep 81-100)
- Semantic weight lambda warmup: 0.01 -> 0.10 over epochs 1-10
- Linear warmup + Cosine Annealing learning rate schedule
"""

import math
from typing import Dict, Any, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler

from .config import TrainConfig


class CurriculumScheduler:
    """
    Manages Teacher Forcing probability and Semantic Loss weight (lambda) across epochs.
    """
    def __init__(
        self,
        total_epochs: int = 100,
        stage1_end: Optional[int] = None,
        stage2_end: Optional[int] = None,
        semantic_warmup_epochs: int = 10,
        semantic_weight_start: float = 0.01,
        semantic_weight_max: float = 0.10
    ):
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        self.total_epochs = total_epochs
        default_first = 1 if total_epochs < 3 else min(total_epochs - 2, max(1, round(0.2 * total_epochs)))
        default_second = default_first if total_epochs < 3 else min(total_epochs - 1, max(default_first + 1, round(0.8 * total_epochs)))
        self.stage1_end = default_first if stage1_end is None else min(stage1_end, total_epochs)
        self.stage2_end = default_second if stage2_end is None else min(stage2_end, total_epochs)
        if not 1 <= self.stage1_end <= self.stage2_end <= total_epochs:
            raise ValueError("Invalid teacher-forcing stage boundaries")
        self.semantic_warmup_epochs = min(semantic_warmup_epochs, total_epochs)
        self.semantic_weight_start = semantic_weight_start
        self.semantic_weight_max = semantic_weight_max

    def get_stage_info(self, epoch: int) -> Dict[str, Any]:
        """
        epoch is 1-indexed (1 to total_epochs).
        """
        # Teacher Forcing probability
        if epoch <= self.stage1_end:
            stage_name = "teacher_forcing"
            tf_p = 1.0
        elif epoch <= self.stage2_end:
            stage_name = "gradual_transition"
            progress = (epoch - self.stage1_end) / max(1, (self.stage2_end - self.stage1_end))
            tf_p = max(0.0, min(1.0, 1.0 - progress))
        else:
            stage_name = "autonomous"
            tf_p = 0.0

        # Semantic loss weight lambda warmup
        if epoch <= self.semantic_warmup_epochs:
            progress = (epoch - 1) / max(1, (self.semantic_warmup_epochs - 1))
            lambda_sem = self.semantic_weight_start + progress * (self.semantic_weight_max - self.semantic_weight_start)
        else:
            lambda_sem = self.semantic_weight_max

        return {
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "stage_name": stage_name,
            "tf_prob": float(tf_p),
            "semantic_weight": float(lambda_sem),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_epochs": self.total_epochs,
            "stage1_end": self.stage1_end,
            "stage2_end": self.stage2_end,
            "semantic_warmup_epochs": self.semantic_warmup_epochs,
            "semantic_weight_start": self.semantic_weight_start,
            "semantic_weight_max": self.semantic_weight_max,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.total_epochs = state["total_epochs"]
        self.stage1_end = state["stage1_end"]
        self.stage2_end = state["stage2_end"]
        self.semantic_warmup_epochs = state.get("semantic_warmup_epochs", 10)
        self.semantic_weight_start = state.get("semantic_weight_start", 0.01)
        self.semantic_weight_max = state.get("semantic_weight_max", 0.10)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-5
) -> _LRScheduler:
    """
    Builds Warmup + Cosine Annealing learning rate scheduler.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_epochs = min(warmup_epochs, max(0, total_epochs - 1))
    min_ratio = min(1.0, min_lr / max(1e-8, base_lr))

    def lr_lambda(epoch: int) -> float:
        # epoch is 0-indexed in PyTorch
        current_ep = epoch + 1
        if current_ep <= warmup_epochs:
            return min_ratio + (1.0 - min_ratio) * (current_ep / max(1, warmup_epochs))
        else:
            decay_epochs = total_epochs - warmup_epochs
            current_decay_ep = min(decay_epochs, max(0, current_ep - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * current_decay_ep / max(1, decay_epochs)))
            return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
