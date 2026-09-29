"""
Dataset, Augmentation, and Jigsaw Problem Generator for v3 (Tiny ImageNet):
- BFS expansion order from center seed (2, 2)
- Replicate padding (64x64 -> 65x65) and 13x13 patch slicing
- TwoViewTransform for VICReg self-supervised training
- Single-view evaluation transform for deterministic validation / test
"""

import os
from typing import Tuple, List, Dict, Any, Optional, Callable
import random
import hashlib
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

from .config import DataConfig, AugmentationConfig


def generate_bfs_order(grid_size: int = 5, seed_coord: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """
    Generates deterministic BFS expansion order starting from seed_coord.
    Traverses neighbors strictly in order: Top (-1, 0), Right (0, 1), Bottom (1, 0), Left (0, -1).
    Returns list of (grid_size * grid_size - 1) coordinates (excluding the seed).
    """
    seed_coord = seed_coord or (grid_size // 2, grid_size // 2)
    sr, sc = seed_coord
    if not (0 <= sr < grid_size and 0 <= sc < grid_size):
        raise ValueError("Seed coordinate outside the grid")
    visited = {(sr, sc)}
    queue = [(sr, sc)]
    order: List[Tuple[int, int]] = []
    # Neighbor order: Top, Right, Bottom, Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:
        r, c = queue.pop(0)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in visited:
                visited.add((nr, nc))
                order.append((nr, nc))
                queue.append((nr, nc))

    assert len(order) == grid_size * grid_size - 1, f"Expected {grid_size * grid_size - 1} slots, got {len(order)}"
    return order


def pad_and_slice_image(
    img: torch.Tensor,
    grid_size: int = 5,
    patch_size: Optional[int] = None,
    pad: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pads image using replicate mode and slices into grid_size x grid_size non-overlapping patches.
    img: (3, H, W)
    pad: (left, right, top, bottom)
    Returns:
        patches: (K, 3, patch_size, patch_size) in raster order (r*G + c)
        orig_shape: (H, W)
    """
    orig_h, orig_w = img.shape[1], img.shape[2]
    if grid_size not in (3, 5, 7) or orig_h != orig_w:
        raise ValueError("Expected a square image and grid_size in (3, 5, 7)")
    patch_size = patch_size or (orig_h + grid_size - 1) // grid_size
    pad = pad if pad is not None else (0, patch_size * grid_size - orig_w, 0, patch_size * grid_size - orig_h)
    if orig_w + pad[0] + pad[1] != grid_size * patch_size or orig_h + pad[2] + pad[3] != grid_size * patch_size:
        raise ValueError("Padding and patch size do not cover the image exactly")
    # F.pad expects (left, right, top, bottom)
    padded = F.pad(img.unsqueeze(0), pad, mode="replicate").squeeze(0)  # (3, 65, 65)

    patches = []
    for r in range(grid_size):
        for c in range(grid_size):
            top = r * patch_size
            left = c * patch_size
            patch = padded[:, top:top + patch_size, left:left + patch_size]
            patches.append(patch)

    patches_tensor = torch.stack(patches, dim=0)  # (K, 3, 13, 13)
    return patches_tensor, (orig_h, orig_w)


def assemble_patches(
    patches: torch.Tensor,
    slot_to_cand: torch.Tensor,
    grid_size: int = 5,
    patch_size: Optional[int] = None,
    pad: Optional[Tuple[int, int, int, int]] = None,
    target_shape: Optional[Tuple[int, int]] = (64, 64)
) -> torch.Tensor:
    """
    Reconstructs (3, H, W) full image from placed candidate patches.
    patches: (K, 3, patch_size, patch_size)
    slot_to_cand: (grid_size, grid_size)
    target_shape: (H, W)
    """
    patch_size = patch_size or patches.shape[-1]
    pad = pad if pad is not None else (0, 0, 0, 0)
    full_h = grid_size * patch_size
    full_w = grid_size * patch_size
    assembled = torch.zeros((3, full_h, full_w), dtype=patches.dtype, device=patches.device)

    for r in range(grid_size):
        for c in range(grid_size):
            cand_idx = int(slot_to_cand[r, c])
            if cand_idx >= 0:
                top = r * patch_size
                left = c * patch_size
                assembled[:, top:top + patch_size, left:left + patch_size] = patches[cand_idx]

    if target_shape is not None:
        left_pad, right_pad, top_pad, bottom_pad = pad
        orig_h, orig_w = target_shape
        return assembled[:, top_pad:top_pad + orig_h, left_pad:left_pad + orig_w]
    return assembled


def make_puzzle_dict(
    img_tensor: torch.Tensor,
    grid_size: int = 5,
    seed_coord: Optional[Tuple[int, int]] = None,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Slices image into 25 patches, permutes candidates, sets center seed at (2, 2).
    """
    sampler = rng if rng is not None else random
    K = grid_size * grid_size

    # Original patches in raster order
    orig_patches, orig_shape = pad_and_slice_image(img_tensor, grid_size=grid_size)

    # Random permutation of candidates: cand_perm[i] = original raster slot of candidate i
    cand_perm = list(range(K))
    sampler.shuffle(cand_perm)

    candidates = orig_patches[cand_perm]  # (K, 3, 13, 13)

    # Invert permutation: inv_perm[s] = candidate index for raster slot s
    inv_perm = [0] * K
    for i, s in enumerate(cand_perm):
        inv_perm[s] = i

    target_mapping = torch.tensor(inv_perm, dtype=torch.long).view(grid_size, grid_size)

    seed_coord = seed_coord or (grid_size // 2, grid_size // 2)
    sr, sc = seed_coord
    if not (0 <= sr < grid_size and 0 <= sc < grid_size):
        raise ValueError("Seed coordinate outside the grid")
    seed_cand = int(target_mapping[sr, sc])

    return {
        "candidates": candidates,              # (K, 3, 13, 13)
        "seed_cand": seed_cand,                # int
        "seed_coord": (sr, sc),                # (2, 2)
        "target_mapping": target_mapping,      # (5, 5)
        "orig_shape": orig_shape,              # (64, 64)
        "cand_perm": cand_perm,                # List[int]
    }


class TwoViewTransform:
    """
    Applies two independent stochastic augmentations to an input PIL image.
    """
    def __init__(self, aug_config: Optional[AugmentationConfig] = None):
        cfg = aug_config or AugmentationConfig()
        self.transform = T.Compose([
            T.RandomResizedCrop(64, scale=cfg.crop_scale),
            T.RandomHorizontalFlip(p=cfg.hflip_p),
            T.RandomApply([
                T.ColorJitter(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue
                )
            ], p=cfg.color_jitter_p),
            T.RandomGrayscale(p=cfg.grayscale_p),
            T.ToTensor(),
            T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std)
        ])

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(img), self.transform(img)


class EvalTransform:
    """
    Deterministic evaluation transform: clean resize, ToTensor, Normalize.
    """
    def __init__(self, aug_config: Optional[AugmentationConfig] = None):
        cfg = aug_config or AugmentationConfig()
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std)
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class TinyImageNetDualDataset(Dataset):
    """
    Dual-view training dataset for Tiny ImageNet:
    Each image produces two augmented views, each packaged as an independent 5x5 jigsaw problem.
    """
    def __init__(
        self,
        root_dir: str,
        records: List[Dict[str, Any]],
        grid_size: int = 5,
        seed_coord: Optional[Tuple[int, int]] = None,
        aug_config: Optional[AugmentationConfig] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.records = records
        self.grid_size = grid_size
        self.seed_coord = seed_coord or (grid_size // 2, grid_size // 2)
        self.transform = TwoViewTransform(aug_config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = os.path.join(self.root_dir, rec["rel_path"])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            view1_tensor, view2_tensor = self.transform(img)

        # Independent candidate shuffles for view 1 and view 2
        p1 = make_puzzle_dict(view1_tensor, grid_size=self.grid_size, seed_coord=self.seed_coord)
        p2 = make_puzzle_dict(view2_tensor, grid_size=self.grid_size, seed_coord=self.seed_coord)

        return {
            "view1": p1,
            "view2": p2,
            "class_idx": rec["class_idx"],
            "wnid": rec["wnid"],
            "rel_path": rec["rel_path"]
        }


class TinyImageNetSingleDataset(Dataset):
    """
    Single-view deterministic dataset for validation and representation evaluation.
    Candidate permutations are deterministically seeded by index so all checkpoints
    face the identical puzzle problems.
    """
    def __init__(
        self,
        root_dir: str,
        records: List[Dict[str, Any]],
        grid_size: int = 5,
        seed_coord: Optional[Tuple[int, int]] = None,
        base_seed: int = 42,
        aug_config: Optional[AugmentationConfig] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.records = records
        self.grid_size = grid_size
        self.seed_coord = seed_coord or (grid_size // 2, grid_size // 2)
        self.base_seed = base_seed
        self.transform = EvalTransform(aug_config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = os.path.join(self.root_dir, rec["rel_path"])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img)

        # Deterministic shuffle for validation reproducibility
        path_seed = int.from_bytes(hashlib.sha256(rec["rel_path"].encode()).digest()[:8], "little")
        rng = random.Random(self.base_seed + path_seed)
        p = make_puzzle_dict(img_tensor, grid_size=self.grid_size, seed_coord=self.seed_coord, rng=rng)

        return {
            "candidates": p["candidates"],
            "seed_cand": p["seed_cand"],
            "seed_coord": p["seed_coord"],
            "target_mapping": p["target_mapping"],
            "cand_perm": p["cand_perm"],
            "orig_shape": p["orig_shape"],
            "raw_img": img_tensor,
            "class_idx": rec["class_idx"],
            "wnid": rec["wnid"],
            "rel_path": rec["rel_path"]
        }


def collate_two_views(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch collator for dual-view training.
    """
    def pack_view(key: str):
        candidates = torch.stack([item[key]["candidates"] for item in batch], dim=0)
        seed_cands = torch.tensor([item[key]["seed_cand"] for item in batch], dtype=torch.long)
        seed_coords = torch.tensor([item[key]["seed_coord"] for item in batch], dtype=torch.long)
        target_mappings = torch.stack([item[key]["target_mapping"] for item in batch], dim=0)
        cand_perms = [item[key]["cand_perm"] for item in batch]
        return {
            "candidates": candidates,          # (B, 25, 3, 13, 13)
            "seed_cand": seed_cands,           # (B,)
            "seed_coord": seed_coords,         # (B, 2)
            "target_mapping": target_mappings, # (B, 5, 5)
            "cand_perm": cand_perms,
        }

    return {
        "view1": pack_view("view1"),
        "view2": pack_view("view2"),
        "class_idx": torch.tensor([item["class_idx"] for item in batch], dtype=torch.long),
        "wnids": [item["wnid"] for item in batch],
        "rel_paths": [item["rel_path"] for item in batch]
    }


def collate_single_view(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch collator for single-view evaluation / validation.
    """
    candidates = torch.stack([item["candidates"] for item in batch], dim=0)
    seed_cands = torch.tensor([item["seed_cand"] for item in batch], dtype=torch.long)
    seed_coords = torch.tensor([item["seed_coord"] for item in batch], dtype=torch.long)
    target_mappings = torch.stack([item["target_mapping"] for item in batch], dim=0)
    cand_perms = [item["cand_perm"] for item in batch]
    raw_imgs = torch.stack([item["raw_img"] for item in batch], dim=0)
    class_indices = torch.tensor([item["class_idx"] for item in batch], dtype=torch.long)

    return {
        "candidates": candidates,              # (B, 25, 3, 13, 13)
        "seed_cand": seed_cands,               # (B,)
        "seed_coord": seed_coords,             # (B, 2)
        "target_mapping": target_mappings,     # (B, 5, 5)
        "cand_perm": cand_perms,
        "raw_img": raw_imgs,                   # (B, 3, 64, 64)
        "class_idx": class_indices,            # (B,)
        "wnids": [item["wnid"] for item in batch],
        "rel_paths": [item["rel_path"] for item in batch]
    }
