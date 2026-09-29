"""
局部十字融合模块 (LocalFusion)
在 3x3 十字窗口 (目标块 + 上下左右有效邻居) 内执行局部注意力融合。
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTransformerBlock(nn.Module):
    """
    单层 Pre-LN Transformer 模块:
    dim: 96, heads: 4, mlp_ratio: 2
    """
    def __init__(self, dim: int = 96, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, seq_len, dim)
        key_padding_mask: (B, seq_len), True 表示该位置为无效填充
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LocalFusion(nn.Module):
    """
    在 3x3 十字窗口 (目标块 + 上下左右有效邻居) 内执行局部融合:
    1. 掩码深度可分离卷积 (Depthwise + Pointwise Conv)
    2. 局部十字注意力 (1 层 Local Transformer Block)
    """
    def __init__(self, dim: int = 96, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.dim = dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gn = nn.GroupNorm(12, dim)
        self.act = nn.GELU()

        self.local_transformer = LocalTransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        grid_features: torch.Tensor, # (B, dim, g, g)
        grid_mask: torch.Tensor,     # (B, 1, g, g)
        target_r: int,
        target_c: int,
        valid_coords: Optional[set] = None
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        对已填入中心 (target_r, target_c) 及其有效正交邻居执行局部融合，返回更新后的特征映射字典。
        """
        B, C, g, _ = grid_features.shape

        r_start = max(0, target_r - 1)
        r_end = min(g, target_r + 2)
        c_start = max(0, target_c - 1)
        c_end = min(g, target_c + 2)

        local_feat = torch.zeros((B, C, 3, 3), device=grid_features.device, dtype=grid_features.dtype)
        local_mask = torch.zeros((B, 1, 3, 3), device=grid_features.device, dtype=grid_mask.dtype)

        dst_r_start = 1 - (target_r - r_start)
        dst_r_end = dst_r_start + (r_end - r_start)
        dst_c_start = 1 - (target_c - c_start)
        dst_c_end = dst_c_start + (c_end - c_start)

        local_feat[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = grid_features[:, :, r_start:r_end, c_start:c_end]
        local_mask[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = grid_mask[:, :, r_start:r_end, c_start:c_end]

        cross_pattern = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]
        ], device=grid_features.device).view(1, 1, 3, 3)

        effective_mask = local_mask * cross_pattern # (B, 1, 3, 3)

        # 1. 深度可分离卷积
        masked_input = local_feat * effective_mask
        counts = F.conv2d(effective_mask, torch.ones(1, 1, 3, 3,
                         device=grid_features.device, dtype=grid_features.dtype), padding=1)
        spatial = self.dw_conv(masked_input) * (9.0 / counts.clamp_min(1))
        conv_out = self.act(self.gn(self.pw_conv(spatial)))
        local_feat = local_feat + conv_out * effective_mask

        # 2. 局部十字注意力
        cross_coords = [(1, 1), (0, 1), (1, 2), (2, 1), (1, 0)]
        tokens = []
        token_masks = []
        for r_rel, c_rel in cross_coords:
            tok = local_feat[:, :, r_rel, c_rel]
            tok_m = effective_mask[:, 0, r_rel, c_rel]
            tokens.append(tok)
            token_masks.append(tok_m)

        seq_tokens = torch.stack(tokens, dim=1) # (B, 5, C)
        pad_mask = torch.stack(token_masks, dim=1) < 0.5 # (B, 5)

        trans_out = self.local_transformer(seq_tokens, key_padding_mask=pad_mask)
        updated = self.out_norm(trans_out)

        updated_dict = {}
        for index, (r_rel, c_rel) in enumerate(cross_coords):
            real_r = target_r + r_rel - 1
            real_c = target_c + c_rel - 1
            if 0 <= real_r < g and 0 <= real_c < g:
                if valid_coords is not None and (real_r, real_c) in valid_coords:
                    updated_dict[(real_r, real_c)] = updated[:, index]
                elif valid_coords is None and effective_mask[:, 0, r_rel, c_rel].any():
                    updated_dict[(real_r, real_c)] = updated[:, index]

        return updated_dict
