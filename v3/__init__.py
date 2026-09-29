"""
v3: Tiny ImageNet Jigsaw Puzzle & Whole-Image Representation Learning Package.
"""

from .config import DataConfig, AugmentationConfig, ModelConfig, TrainConfig
from .cnn_encoder import PatchCNNEncoder
from .model import JigsawSolverV3, VirtualPatchModule
from .vicreg import VICRegProjector, VICRegLoss
from .solver import solve_batch, solve_single, compute_puzzle_accuracy
from .trainer import JigsawTrainerV3
