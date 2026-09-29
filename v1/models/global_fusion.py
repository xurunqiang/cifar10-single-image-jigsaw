"""
全局网格融合模块 (GlobalFusion)
网格深度卷积 + 2D 位置编码 + 双层全局 Transformer
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalFusion(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        grid_size: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gn = nn.GroupNorm(12, dim)
        self.act = nn.GELU()

        self.pos_embed = nn.Parameter(torch.zeros(1, dim, grid_size, grid_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, grid_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: grid_features (B, 96, g, g)
        输出:
            z_global: (B, 96) 全图池化语义向量
            slot_tokens: (B, g*g, 96) 经过全局 Transformer 上下文增强的每个槽位特征
        """
        B, C, H, W = grid_features.shape
        assert H == self.grid_size and W == self.grid_size, f"输入网格大小需为 {self.grid_size}x{self.grid_size}"

        conv_out = self.act(self.gn(self.pw_conv(self.dw_conv(grid_features))))
        h = grid_features + conv_out
        h = h + self.pos_embed

        seq = h.flatten(2).permute(0, 2, 1) # (B, g*g, 96)
        seq_out = self.transformer(seq)     # (B, g*g, 96)
        global_repr = self.out_norm(seq_out.mean(dim=1)) # (B, 96)
        return global_repr, seq_out
