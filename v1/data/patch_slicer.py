"""
单图切片与还原工具 (PatchSlicer)
对 CIFAR-10 (32x32) 进行非对称反射填充、切片以及拼图还原重构。
彻底移除原版的边缘提取与法向梯度。
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from v1.config import SlicerConfig


class PatchSlicer:
    """
    针对 CIFAR-10 (32x32) 的网格切片与还原工具。
    支持 3x3 (P=11), 5x5 (P=7), 7x7 (P=5)。
    """
    def __init__(self, grid_size: int = 3):
        self.grid_size = grid_size
        
        if grid_size == 3:
            self.padded_size = (33, 33)
            self.patch_size = 11
            self.pad = (0, 1, 0, 1) # (left, right, top, bottom)
        elif grid_size == 5:
            self.padded_size = (35, 35)
            self.patch_size = 7
            self.pad = (1, 2, 1, 2)
        elif grid_size == 7:
            self.padded_size = (35, 35)
            self.patch_size = 5
            self.pad = (1, 2, 1, 2)
        else:
            raise ValueError(f"不支持的网格尺寸: {grid_size}，仅支持 3, 5, 7")

        self.center_coord = (grid_size // 2, grid_size // 2)

    def slice_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        输入: img (B, 3, 32, 32)
        输出: patches (B, grid_size, grid_size, 3, P, P)
        """
        B, C, H, W = img.shape
        assert H == 32 and W == 32, f"输入必须为 32x32，当前为 {H}x{W}"

        # 1. 反射填充至指定尺寸 (33x33 或 35x35)
        padded_img = F.pad(img, self.pad, mode='reflect')

        # 2. unfold 切片
        P = self.patch_size
        patches = padded_img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches

    def unslice_image(self, patches: torch.Tensor, crop_to_32: bool = True) -> torch.Tensor:
        """
        将组装好的 patches (B, g, g, 3, P, P) 拼接还原为整图 (B, 3, H, W)。
        如果 crop_to_32=True，裁剪掉外围填充边缘，恢复原始 32x32 图像。
        """
        B, g, _, C, P, _ = patches.shape
        assert g == self.grid_size and P == self.patch_size

        # 拼装回 (B, 3, g * P, g * P)
        full_img = patches.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(
            B, C, g * P, g * P
        )

        if crop_to_32:
            left, right, top, bottom = self.pad
            h_end = full_img.shape[2] - bottom if bottom > 0 else full_img.shape[2]
            w_end = full_img.shape[3] - right if right > 0 else full_img.shape[3]
            return full_img[:, :, top:h_end, left:w_end]

        return full_img
