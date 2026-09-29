"""
Dataset, image slicing, and problem generation for v2 jigsaw puzzle solver.
"""

import math
import random
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms

from .config import GridConfig


def get_grid_specs(grid_size: int) -> Dict[str, Any]:
    specs = GridConfig.SPECS
    if grid_size not in specs:
        raise ValueError(f"Unsupported grid_size: {grid_size}. Supported: {list(specs.keys())}")
    return specs[grid_size]


def pad_and_slice_image(img: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad image according to GridConfig and slice into grid_size x grid_size patches.
    Args:
        img: (3, H, W) tensor in range [0, 1] or normalized.
        grid_size: 3, 5, or 7.
    Returns:
        patches: (K, 3, P, P) where K = grid_size * grid_size.
        orig_shape: (H, W).
    """
    spec = get_grid_specs(grid_size)
    pad = spec["pad"]  # (left, right, top, bottom)
    patch_size = spec["patch_size"]

    # F.pad expects (left, right, top, bottom)
    padded = F.pad(img, pad, mode="replicate")
    _, pad_h, pad_w = padded.shape
    assert pad_h == grid_size * patch_size and pad_w == grid_size * patch_size, \
        f"Padded shape {(pad_h, pad_w)} does not match grid_size * patch_size = {grid_size * patch_size}"

    patches = []
    for r in range(grid_size):
        for c in range(grid_size):
            top = r * patch_size
            left = c * patch_size
            patch = padded[:, top:top + patch_size, left:left + patch_size]
            patches.append(patch)

    patches_tensor = torch.stack(patches, dim=0)  # (K, 3, P, P)
    return patches_tensor, (img.shape[1], img.shape[2])


def assemble_patches(
    patches: torch.Tensor,
    slot_to_cand: torch.Tensor,
    grid_size: int,
    target_shape: Optional[Tuple[int, int]] = (32, 32)
) -> torch.Tensor:
    """
    Reconstruct the full image from patches placed at each slot.
    Args:
        patches: (K, 3, P, P) tensor of candidate patches.
        slot_to_cand: (grid_size, grid_size) tensor or array of candidate indices.
        grid_size: int (3, 5, 7).
        target_shape: (H, W) to unpad/crop back to original size, default (32, 32).
    Returns:
        assembled: (3, H, W) reconstructed image tensor.
    """
    spec = get_grid_specs(grid_size)
    patch_size = spec["patch_size"]
    pad = spec["pad"]  # (left, right, top, bottom)

    full_h = grid_size * patch_size
    full_w = grid_size * patch_size
    assembled_padded = torch.zeros((3, full_h, full_w), dtype=patches.dtype, device=patches.device)

    for r in range(grid_size):
        for c in range(grid_size):
            cand_idx = int(slot_to_cand[r, c])
            if cand_idx >= 0:
                patch = patches[cand_idx]
                top = r * patch_size
                left = c * patch_size
                assembled_padded[:, top:top + patch_size, left:left + patch_size] = patch

    # Crop out padding to restore target_shape
    if target_shape is not None:
        orig_h, orig_w = target_shape
        left_pad, _, top_pad, _ = pad
        return assembled_padded[:, top_pad:top_pad + orig_h, left_pad:left_pad + orig_w]
    return assembled_padded


class JigsawProblemGenerator:
    """
    Creates a puzzle problem from an original image:
    1. Slices into K patches.
    2. Shuffles patches into candidate pool via permutation.
    3. Selects seed candidate & true coordinate.
    4. Provides target_mapping for ground truth supervision/evaluation.
    """
    @staticmethod
    def generate(
        img: torch.Tensor,
        grid_size: int,
        seed_mode: str = "random",
        seed_coord: Optional[Tuple[int, int]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Args:
            img: (3, H, W)
            grid_size: 3, 5, or 7
            seed_mode: "random", "center", or "custom"
            seed_coord: Tuple (r, c) if seed_mode == "custom"
            rng: random.Random instance for reproducibility
        """
        sampler = rng if rng is not None else random
        K = grid_size * grid_size

        # Slices in original raster order: slot s = r * grid_size + c
        orig_patches, orig_shape = pad_and_slice_image(img, grid_size)

        # Generate random permutation of candidates: cand_perm[i] is the original slot of candidate i
        cand_perm = list(range(K))
        sampler.shuffle(cand_perm)

        # Candidates tensor: cand_perm[i] tells which original patch is in candidate index i
        candidates = orig_patches[cand_perm]  # (K, 3, P, P)

        # target_mapping: for each grid slot (r, c), which candidate index i corresponds to it?
        # Let s = r * grid_size + c. Since candidate i contains orig_patches[cand_perm[i]],
        # if cand_perm[i] == s, then candidate i belongs to slot (r, c).
        # Hence target_mapping[r, c] = cand_perm.index(s).
        inv_perm = [0] * K
        for i, s in enumerate(cand_perm):
            inv_perm[s] = i

        target_mapping = torch.tensor(inv_perm, dtype=torch.long).view(grid_size, grid_size)

        # Determine seed coordinate
        if seed_mode == "center":
            sr, sc = grid_size // 2, grid_size // 2
        elif seed_mode == "custom":
            assert seed_coord is not None
            sr, sc = seed_coord
        elif seed_mode == "random":
            sr = sampler.randint(0, grid_size - 1)
            sc = sampler.randint(0, grid_size - 1)
        else:
            raise ValueError(f"Unknown seed_mode: {seed_mode}")

        if not (0 <= sr < grid_size and 0 <= sc < grid_size):
            raise ValueError(f"Seed coordinate {(sr, sc)} outside {grid_size}x{grid_size} grid")
        seed_cand = int(target_mapping[sr, sc])

        return {
            "candidates": candidates,              # (K, 3, P, P)
            "seed_cand": seed_cand,                # int
            "seed_coord": (sr, sc),                # (int, int)
            "target_mapping": target_mapping,      # (grid_size, grid_size)
            "orig_shape": orig_shape,              # (H, W)
            "cand_perm": cand_perm,                # List[int], original slot for each candidate
        }


class CIFAR10JigsawDataset(Dataset):
    """
    CIFAR-10 Jigsaw Dataset for v2.
    Supports random seed selection per epoch during training,
    and fixed deterministic configuration during validation/testing.
    """
    def __init__(
        self,
        root: str,
        grid_size: int = 3,
        train: bool = True,
        seed_mode: str = "random",
        seed_coord: Optional[Tuple[int, int]] = None,
        base_seed: int = 42,
        download: bool = False,
        vary_by_epoch: Optional[bool] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.train = train
        self.vary_by_epoch = train if vary_by_epoch is None else vary_by_epoch
        self.seed_mode = seed_mode
        self.seed_coord = seed_coord
        self.base_seed = base_seed
        self.epoch = 0

        # Normalization standard for CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        self.cifar = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, label = self.cifar[idx]

        # For validation: deterministic RNG per idx
        # For training: RNG depends on base_seed + epoch * 100000 + idx
        if not self.vary_by_epoch:
            sample_seed = self.base_seed + idx * 7919
        else:
            sample_seed = self.base_seed + self.epoch * 1000003 + idx * 7919

        rng = random.Random(sample_seed)

        problem = JigsawProblemGenerator.generate(
            img=img,
            grid_size=self.grid_size,
            seed_mode=self.seed_mode,
            seed_coord=self.seed_coord,
            rng=rng
        )
        problem["img_idx"] = idx
        problem["expansion_seed"] = sample_seed + 104729
        problem["class_label"] = label
        return problem


def collate_jigsaw(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch of jigsaw problems into tensors.
    """
    candidates = torch.stack([item["candidates"] for item in batch], dim=0)  # (B, K, 3, P, P)
    seed_cands = torch.tensor([item["seed_cand"] for item in batch], dtype=torch.long)  # (B,)
    seed_coords = torch.tensor([item["seed_coord"] for item in batch], dtype=torch.long)  # (B, 2)
    target_mappings = torch.stack([item["target_mapping"] for item in batch], dim=0)  # (B, G, G)
    img_indices = [item["img_idx"] for item in batch]

    return {
        "candidates": candidates,
        "seed_cand": seed_cands,
        "seed_coord": seed_coords,
        "target_mapping": target_mappings,
        "img_indices": img_indices,
        "expansion_seeds": [item.get("expansion_seed", 104729 + item["img_idx"] * 7919) for item in batch],
    }


def set_dataset_epoch(dataset: Dataset, epoch: int):
    """Propagate the epoch through Subset wrappers (workers must be nonpersistent)."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def build_train_val_datasets(root, grid_size=3, seed_mode="random", base_seed=42,
                             seed_coord=None, max_train_samples=None, max_val_samples=None):
    """Split the official training set; reserve the official test set for final evaluation."""
    train = CIFAR10JigsawDataset(root, grid_size, train=True, seed_mode=seed_mode,
                                seed_coord=seed_coord, base_seed=base_seed)
    val = CIFAR10JigsawDataset(root, grid_size, train=True, seed_mode=seed_mode,
                              seed_coord=seed_coord, base_seed=base_seed, vary_by_epoch=False)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(base_seed)).tolist()
    split = int(len(indices) * 0.9)
    train_indices, val_indices = indices[:split], indices[split:]
    for limit in (max_train_samples, max_val_samples):
        if limit is not None and limit <= 0:
            raise ValueError("Sample limits must be positive")
    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]
    return Subset(train, train_indices), Subset(val, val_indices)
