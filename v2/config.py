"""
Configuration definitions for v2 jigsaw puzzle solver.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, ClassVar, Optional


class GridConfig:
    """
    Grid slicing and padding specifications for CIFAR-10 (32x32).
    pad: (left, right, top, bottom)
    """
    SPECS: Dict[int, Dict[str, Any]] = {
        3: {
            "padded_size": (33, 33),
            "patch_size": 11,
            "pad": (0, 1, 0, 1),
        },
        5: {
            "padded_size": (35, 35),
            "patch_size": 7,
            "pad": (1, 2, 1, 2),
        },
        7: {
            "padded_size": (35, 35),
            "patch_size": 5,
            "pad": (1, 2, 1, 2),
        },
    }


@dataclass
class ModelConfig:
    grid_size: int = 3
    content_dim: int = 96
    local_heads: int = 4
    global_layers: int = 2
    global_heads: int = 4
    mlp_ratio: int = 2
    temperature: float = 0.1
    mode: str = "both"  # "both", "local_only", "global_only"

    def __post_init__(self):
        if self.grid_size not in GridConfig.SPECS:
            raise ValueError("grid_size must be 3, 5, or 7")
        if self.content_dim <= 0 or self.content_dim % 12:
            raise ValueError("content_dim must be positive and divisible by 12 for GroupNorm")
        if any(h <= 0 or self.content_dim % h for h in (self.local_heads, self.global_heads)):
            raise ValueError("Attention heads must divide content_dim")
        if self.temperature <= 0 or self.global_layers < 1 or self.mlp_ratio < 1:
            raise ValueError("Temperature, layer count and MLP ratio must be positive")
        if self.mode not in ("both", "local_only", "global_only"):
            raise ValueError("Unknown semantic branch mode")


@dataclass
class TrainConfig:
    data_dir: str = "/home/cjc/桌面/myidea/data/cifar10"
    save_dir_prefix: str = "checkpoints/v2"
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    epochs: int = 100
    expansion_strategy: str = "random_frontier"  # "random_frontier" or "bfs"
    seed_selection: str = "random"  # "random", "center", or "(r,c)"
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    seed_coord: Optional[Tuple[int, int]] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.warmup_epochs < 0:
            raise ValueError("Invalid training duration, batch size, or learning rate")
        if self.seed_selection not in ("random", "center", "custom"):
            raise ValueError("Unknown seed selection mode")
        if self.seed_selection == "custom" and self.seed_coord is None:
            raise ValueError("Custom seed selection requires seed_coord")

    def get_save_dir(self, grid_size: int) -> str:
        return f"{self.save_dir_prefix}/grid{grid_size}"
