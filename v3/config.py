"""
Configuration definitions for v3: Tiny ImageNet Jigsaw & Whole-Image Representation Learning.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional


@dataclass
class DataConfig:
    root_dir: str = "data/tiny_imagenet/tiny-imagenet-200"
    split_file: str = "data/tiny_imagenet/split_seed42.json"
    grid_size: int = 5
    image_size: int = 64
    padded_size: int = 65
    patch_size: int = 13
    # F.pad expects (left, right, top, bottom): right=1, bottom=1
    pad: Tuple[int, int, int, int] = (0, 1, 0, 1)
    seed_coord: Tuple[int, int] = (2, 2)
    num_classes: int = 200

    def __post_init__(self):
        if self.grid_size not in (3, 5, 7) or self.image_size != 64:
            raise ValueError("Tiny ImageNet requires image_size=64 and grid_size in (3, 5, 7)")
        self.patch_size = (self.image_size + self.grid_size - 1) // self.grid_size
        self.padded_size = self.patch_size * self.grid_size
        extra = self.padded_size - self.image_size
        self.pad = (0, extra, 0, extra)
        self.seed_coord = (self.grid_size // 2, self.grid_size // 2)


@dataclass
class AugmentationConfig:
    crop_scale: Tuple[float, float] = (0.6, 1.0)
    hflip_p: float = 0.5
    color_jitter_p: float = 0.8
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1
    grayscale_p: float = 0.2
    norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class ModelConfig:
    grid_size: int = 5
    content_dim: int = 256
    local_layers: int = 2
    local_heads: int = 8
    global_layers: int = 4
    global_heads: int = 8
    global_ffn_dim: int = 1024
    pre_ln: bool = True
    gradient_checkpointing: bool = True
    temperature: float = 0.1
    proj_hidden_dim: int = 512
    proj_out_dim: int = 512
    cnn_stages: Tuple[int, int, int] = (64, 128, 256)

    def __post_init__(self):
        self.cnn_stages = tuple(self.cnn_stages)
        if self.grid_size not in (3, 5, 7) or not self.pre_ln:
            raise ValueError("Supported grids are 3/5/7; v3 uses Pre-LN")
        if self.content_dim <= 0 or any(h <= 0 or self.content_dim % h for h in (self.local_heads, self.global_heads)):
            raise ValueError("Attention heads must divide positive content_dim")
        if self.temperature <= 0 or min(self.local_layers, self.global_layers, self.global_ffn_dim, self.proj_hidden_dim, self.proj_out_dim) <= 0:
            raise ValueError("Layers, dimensions and temperature must be positive")
        if len(self.cnn_stages) != 3 or min(self.cnn_stages) <= 0:
            raise ValueError("cnn_stages must contain three positive channel counts")


@dataclass
class TrainConfig:
    batch_size: int = 256  # 256 raw images -> 512 views per step
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    epochs: int = 100
    min_lr: float = 1e-5
    grad_clip: float = 1.0

    # Semantic Loss (VICReg) settings
    semantic_weight_start: float = 0.01
    semantic_weight_max: float = 0.1
    semantic_warmup_epochs: int = 10
    sim_weight: float = 25.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    var_threshold: float = 1.0
    var_eps: float = 1e-4

    # Jigsaw strategy
    expansion_strategy: str = "bfs"
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    save_dir: str = "checkpoints/v3"
    feature_eval_interval: int = 5
    feature_reference_samples: int = 10000
    feature_validation_samples: int = 10000

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid epochs, batch_size or num_workers")
        if self.lr <= 0 or self.grad_clip <= 0 or self.expansion_strategy != "bfs":
            raise ValueError("Positive lr/grad_clip and BFS are required")
        if not 0 <= self.semantic_weight_start <= self.semantic_weight_max:
            raise ValueError("Semantic weights must satisfy 0 <= start <= max")
        if self.semantic_weight_max > 0 and self.batch_size < 2:
            raise ValueError("VICReg requires at least two images per batch")
        if self.feature_eval_interval < 0 or min(self.feature_reference_samples, self.feature_validation_samples) < 200:
            raise ValueError("Feature evaluation requires interval >= 0 and at least 200 samples per split")

    def get_save_dir(self) -> str:
        return self.save_dir
