"""
VICReg (Variance-Invariance-Covariance Regularization) for Self-Supervised Learning in v3.
Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022.

Safeguards:
- Variance and Covariance computations are performed in float32 for numerical stability.
- Off-diagonal covariance is normalized by feature dimension D.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig, ModelConfig


class VICRegProjector(nn.Module):
    """
    MLP projector mapping representation Z (256-dim) to projected embedding Y (512-dim).
    Linear(256, 512) -> LayerNorm(512) -> GELU -> Linear(512, 512)
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Returns a 1D tensor containing the off-diagonal elements of square matrix x."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """
    Computes VICReg loss over two augmented views:
    - Invariance: MSE loss between projected embeddings
    - Variance: Hinge loss enforcing std >= 1.0 along each dimension
    - Covariance: Penalizes off-diagonal covariances to decorrelate representations
    """
    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_threshold: float = 1.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_threshold = var_threshold
        self.var_eps = var_eps

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        y1, y2: (B, D) projected representations
        Returns:
            total_loss: scalar tensor with gradient
            stats: dict of unweighted loss components and statistics
        """
        if y1.ndim != 2 or y1.shape != y2.shape or y1.shape[0] < 2:
            raise ValueError("VICReg needs equal (B, D) views with B >= 2")
        B, D = y1.shape
        # Casting alone is insufficient: an outer autocast would cast covariance
        # matmul back to BF16. Disable autocast for the entire loss calculation.
        with torch.autocast(device_type=y1.device.type, enabled=False):
            y1_fp32, y2_fp32 = y1.float(), y2.float()
            sim_loss = F.mse_loss(y1_fp32, y2_fp32)
            std_y1 = torch.sqrt(y1_fp32.var(dim=0) + self.var_eps)
            std_y2 = torch.sqrt(y2_fp32.var(dim=0) + self.var_eps)
            var_loss = (F.relu(self.var_threshold - std_y1).mean()
                        + F.relu(self.var_threshold - std_y2).mean()) / 2
            y1_cent = y1_fp32 - y1_fp32.mean(dim=0)
            y2_cent = y2_fp32 - y2_fp32.mean(dim=0)
            cov_y1 = (y1_cent.T @ y1_cent) / (B - 1)
            cov_y2 = (y2_cent.T @ y2_cent) / (B - 1)
            cov_loss = (off_diagonal(cov_y1).square().sum()
                        + off_diagonal(cov_y2).square().sum()) / D
            total_loss = (self.sim_weight * sim_loss + self.var_weight * var_loss
                          + self.cov_weight * cov_loss)

        stats = {
            "sim_loss": float(sim_loss.detach().item()),
            "var_loss": float(var_loss.detach().item()),
            "cov_loss": float(cov_loss.detach().item()),
            "std_mean": float(((std_y1.mean() + std_y2.mean()) / 2.0).detach().item())
        }

        return total_loss, stats
