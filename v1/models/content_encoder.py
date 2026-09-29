"""
内容卷积编码器 (ContentEncoder)
将单个 Patch (B, 3, P, P) 编码为 96 维内容特征向量。
"""

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(self, content_dim: int = 96):
        super().__init__()
        self.content_dim = content_dim

        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(6, 48)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(48, content_dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(12, content_dim)
        self.act2 = nn.GELU()

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Linear(content_dim * 2 * 2, content_dim)
        self.norm = nn.LayerNorm(content_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x: (B, 3, P, P)
        输出: feat: (B, 96)
        """
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.act2(self.gn2(self.conv2(h)))
        h = self.pool(h)
        h_flat = h.view(h.size(0), -1)
        feat = self.norm(self.proj(h_flat))
        return feat
