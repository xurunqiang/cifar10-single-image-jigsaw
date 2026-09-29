"""
单图拼图复原系统 (v1 版本) - 全局配置文件
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SlicerConfig:
    """
    网格切片配置:
    grid_size -> (padded_size, patch_size, pad_tuple)
    pad_tuple: (left, right, top, bottom)
    """
    GRID_SPECS: Dict[int, Dict] = field(default_factory=lambda: {
        3: {
            "padded_size": (33, 33),
            "patch_size": 11,
            "pad": (0, 1, 0, 1), # CIFAR-10 32x32 -> 33x33 (右、下各补1)
        },
        5: {
            "padded_size": (35, 35),
            "patch_size": 7,
            "pad": (1, 2, 1, 2), # CIFAR-10 32x32 -> 35x35
        },
        7: {
            "padded_size": (35, 35),
            "patch_size": 5,
            "pad": (1, 2, 1, 2), # CIFAR-10 32x32 -> 35x35
        }
    })


@dataclass
class ModelConfig:
    grid_size: int = 3                  # 默认 3x3 网格 (可选 3, 5, 7)
    content_dim: int = 96               # Patch 内容表征向量维度
    local_heads: int = 4                # 局部十字 Transformer 头数
    local_mlp_ratio: int = 2            # 局部 Transformer MLP 扩展率
    global_layers: int = 2              # 全局 Transformer 层数
    global_heads: int = 4               # 全局 Transformer 头数
    global_mlp_ratio: int = 2           # 全局 Transformer MLP 扩展率
    temperature: float = 0.1            # 匹配分数缩放温度


@dataclass
class TrainConfig:
    data_dir: str = "/home/cjc/桌面/myidea/data/cifar10"
    save_dir: str = "./checkpoints"
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    seed: int = 42
    num_workers: int = 4
