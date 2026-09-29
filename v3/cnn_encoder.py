"""
ResNet-based CNN Content Encoder for v3:
- 3 residual stages with channels 64, 128, 256
- Spatial resolutions: 13x13 -> 7x7 -> 4x4
- GroupNorm + GELU activations
- 2x2 adaptive spatial pooling -> Linear(1024, 256) -> LayerNorm(256)
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_norm(channels: int) -> nn.GroupNorm:
    """Returns GroupNorm with at most 32 groups, ensuring channels is divisible by num_groups."""
    num_groups = min(32, channels)
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


class ResidualBlock(nn.Module):
    """
    Standard residual block with GroupNorm and GELU.
    Conv3x3 -> GN -> GELU -> Conv3x3 -> GN -> Residual Add -> GELU.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = get_group_norm(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = get_group_norm(out_channels)
        self.act2 = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act2(h + res)


class PatchCNNEncoder(nn.Module):
    """
    3-stage Residual CNN encoder for 13x13 image patches:
    Input: (B, K, 3, 13, 13) or (N, 3, 13, 13)
    Output: (B, K, 256) or (N, 256)
    """
    def __init__(self, content_dim: int = 256, stages: Tuple[int, int, int] = (64, 128, 256)):
        super().__init__()
        self.content_dim = content_dim
        c1, c2, c3 = stages

        # Initial conv: 3 -> c1, spatial 13x13
        self.conv0 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn0 = get_group_norm(c1)
        self.act0 = nn.GELU()

        # Stage 1: c1 -> c1, 13x13, 2 blocks
        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c1, stride=1),
            ResidualBlock(c1, c1, stride=1),
        )

        # Stage 2: c1 -> c2, 13x13 -> 7x7, 2 blocks
        self.stage2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2),
            ResidualBlock(c2, c2, stride=1),
        )

        # Stage 3: c2 -> c3, 7x7 -> 4x4, 2 blocks
        self.stage3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2),
            ResidualBlock(c3, c3, stride=1),
        )

        # 2x2 adaptive spatial pooling -> 1024 flat -> 256
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Linear(c3 * 2 * 2, content_dim)
        self.norm = nn.LayerNorm(content_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supports both 4D (N, 3, H, W) and 5D (B, K, 3, H, W) inputs.
        """
        orig_shape = x.shape
        if x.dim() == 5:
            B, K, C, H, W = orig_shape
            x = x.view(B * K, C, H, W)
        elif x.dim() == 4:
            B, K = None, None
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {orig_shape}")

        h = self.act0(self.gn0(self.conv0(x)))
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)

        h = self.pool(h)
        h_flat = h.view(h.size(0), -1)
        feat = self.norm(self.proj(h_flat))

        if B is not None and K is not None:
            feat = feat.view(B, K, self.content_dim)

        return feat
