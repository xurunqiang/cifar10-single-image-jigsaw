"""
拼图复原损失与准确率统计 (PuzzleLoss)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PuzzleLoss(nn.Module):
    """
    针对单图拼图还原任务的槽位分类损失。
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        all_logits: torch.Tensor, # (B, S, K)
        all_targets: torch.Tensor # (B, S)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算交叉熵损失、Patch 级别命中率和整图完全拼对率。
        """
        B, S, K = all_logits.shape
        flat_logits = all_logits.reshape(B * S, K)
        flat_targets = all_targets.reshape(B * S)

        loss = self.ce(flat_logits, flat_targets)

        with torch.no_grad():
            preds = all_logits.argmax(dim=-1) # (B, S)
            correct_mask = (preds == all_targets) # (B, S)
            patch_acc = correct_mask.float().mean().item() * 100.0
            puzzle_acc = correct_mask.all(dim=1).float().mean().item() * 100.0

        return loss, {
            "loss": loss.item(),
            "patch_acc": patch_acc,
            "puzzle_acc": puzzle_acc
        }
