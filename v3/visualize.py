"""
Visualization Suite for v3:
- Reconstruction cards: Original vs Shuffled vs Reconstructed (highlighting seed and errors)
- Virtual Patch Attention Heatmaps: 5x5 heatmap of Q_virtual attention weights over placed patches
"""

import os
import json
from PIL import Image
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches_mpl
import torch
import torchvision.transforms as T

from .config import ModelConfig
from .model import JigsawSolverV3
from .dataset import assemble_patches, pad_and_slice_image
from .solver import solve_single


def denormalize_image(tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """
    tensor: (3, H, W) normalized with mean/std
    returns: (H, W, 3) in [0, 1] float
    """
    img = tensor.clone().detach().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def plot_reconstruction_card(
    orig_img: torch.Tensor,
    cand_patches: torch.Tensor,
    seed_cand: int,
    seed_coord: tuple,
    target_mapping: torch.Tensor,
    grid_placed: torch.Tensor,
    attn_weights: torch.Tensor,
    save_path: str,
    title: Optional[str] = None
) -> None:
    """
    Generates a 4-panel visual reconstruction card:
    1. Ground truth original image
    2. Shuffled candidate grid (with seed highlighted)
    3. Reconstructed image (with seed in green, errors in red)
    4. Virtual Patch attention weight heatmap (5x5)
    """
    grid_size = grid_placed.shape[0]
    patch_size = cand_patches.shape[-1]

    # 1. Denormalize original
    img_orig_np = denormalize_image(orig_img)

    # 2. Shuffled candidate board (5x5 raster of candidate patches)
    slot_to_cand_shuffled = torch.arange(grid_size ** 2).view(grid_size, grid_size)
    img_shuffled = assemble_patches(cand_patches, slot_to_cand_shuffled, grid_size=grid_size)
    img_shuffled_np = denormalize_image(img_shuffled)

    # 3. Model reconstructed image
    img_recon = assemble_patches(cand_patches, grid_placed, grid_size=grid_size)
    img_recon_np = denormalize_image(img_recon)

    # 4. Attention weights heatmap
    # attn_weights is (25,) over candidate indices. We map it back to 5x5 grid slots!
    attn_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            cand_idx = int(grid_placed[r, c])
            if cand_idx >= 0:
                attn_grid[r, c] = float(attn_weights[cand_idx].item())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Panel 1: Original
    axes[0].imshow(img_orig_np)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Shuffled
    axes[1].imshow(img_shuffled_np)
    axes[1].set_title(f"Shuffled (Seed Cand: #{seed_cand})", fontsize=12)
    # Highlight seed on shuffled board
    seed_r, seed_c = seed_cand // grid_size, seed_cand % grid_size
    rect = patches_mpl.Rectangle(
        (seed_c * patch_size, seed_r * patch_size), patch_size, patch_size,
        linewidth=2, edgecolor="lime", facecolor="none"
    )
    axes[1].add_patch(rect)
    axes[1].axis("off")

    # Panel 3: Model Reconstructed
    axes[2].imshow(img_recon_np)
    non_seed_correct = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) == seed_coord:
                # Green border for known seed
                rect = patches_mpl.Rectangle(
                    (c * patch_size, r * patch_size), patch_size, patch_size,
                    linewidth=2.5, edgecolor="lime", facecolor="none"
                )
                axes[2].add_patch(rect)
            else:
                is_correct = (grid_placed[r, c] == target_mapping[r, c])
                if is_correct:
                    non_seed_correct += 1
                else:
                    # Red border for error
                    rect = patches_mpl.Rectangle(
                        (c * patch_size, r * patch_size), patch_size, patch_size,
                        linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
                    )
                    axes[2].add_patch(rect)

    acc_pct = (non_seed_correct / float(grid_size ** 2 - 1)) * 100.0
    axes[2].set_title(f"Reconstructed (Acc: {acc_pct:.1f}%)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Virtual Patch Attention Heatmap
    im4 = axes[3].imshow(attn_grid, cmap="viridis", interpolation="nearest")
    axes[3].set_title(r"$Z_{virtual}$ Attention Weights", fontsize=12, fontweight="bold")
    axes[3].set_xticks(range(grid_size))
    axes[3].set_yticks(range(grid_size))
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    if title:
        plt.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction card to: {save_path}")


def render_sample_cards(
    model: JigsawSolverV3,
    dataset,
    device: torch.device,
    output_dir: str = "visualizations/v3",
    num_cards: int = 10,
    failure_only: bool = False,
    selection_seed: int = 3407
) -> List[str]:
    """
    Renders num_cards reconstruction cards for samples from dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    if num_cards < 1:
        raise ValueError("num_cards must be positive")
    selected = []
    grid_size = model.config.grid_size
    for i in np.random.default_rng(selection_seed).permutation(len(dataset)):
        if len(saved_paths) >= num_cards:
            break
        i = int(i)
        sample = dataset[i]
        patches = sample["candidates"].to(device)
        seed_cand = sample["seed_cand"]
        seed_coord = sample["seed_coord"]
        target_mapping = sample["target_mapping"]
        raw_img = sample["raw_img"]

        res = solve_single(
            model=model,
            patches=patches,
            seed_cand=seed_cand,
            seed_coord=seed_coord,
            grid_size=grid_size,
            target_mapping=target_mapping
        )

        if failure_only and res["perfect"] == 1.0:
            continue
        category = "failure" if failure_only else "random"
        card_path = os.path.join(output_dir, f"{category}_{i:05d}_{sample['wnid']}.png")
        selected.append({"index": i, "path": sample["rel_path"], "patch_acc": res["patch_acc"],
                         "selection": category})
        plot_reconstruction_card(
            orig_img=raw_img,
            cand_patches=patches.cpu(),
            seed_cand=seed_cand,
            seed_coord=seed_coord,
            target_mapping=target_mapping,
            grid_placed=res["grid_placed"],
            attn_weights=res["attn_weights"],
            save_path=card_path,
            title=f"Sample {i+1} | Class: {sample['wnid']} | Patch Acc: {res.get('patch_acc', 0)*100:.1f}%"
        )
        saved_paths.append(card_path)

    manifest_path = os.path.join(output_dir, "failure_selection.json" if failure_only else "random_selection.json")
    with open(manifest_path, "w") as handle:
        json.dump({"seed": selection_seed, "requested": num_cards, "selected": selected}, handle, indent=2)
    return saved_paths


def plot_training_curves(history, save_path):
    if not history:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row["epoch"] for row in history]
    for key in ("loss_total", "loss_jigsaw", "loss_vicreg"):
        axes[0].plot(epochs, [row[key] for row in history], label=key)
    for key in ("val_patch_acc", "val_perfect_acc", "val_neighbor_acc"):
        axes[1].plot(epochs, [row.get(key, np.nan) for row in history], label=key)
    measured = [row for row in history if row.get("feature_knn_acc") is not None]
    axes[1].plot([row["epoch"] for row in measured], [100 * row["feature_knn_acc"] for row in measured], "o-", label="Dev virtual 5-NN")
    for key in ("tf_prob", "semantic_weight", "std_mean"):
        axes[2].plot(epochs, [row.get(key, np.nan) for row in history], label=key)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy (%)")
    axes[2].set_title("Curriculum / projected feature std")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_nearest_neighbors(train_data, test_data, root_dir, output_dir, num_queries=6, seed=3407):
    """Show the same deterministic random queries for each representation."""
    os.makedirs(output_dir, exist_ok=True)
    queries = np.random.default_rng(seed).choice(len(test_data["labels"]),
        size=min(num_queries, len(test_data["labels"])), replace=False)
    count = min(5, len(train_data["labels"]))
    selections = {}
    for feature in ("raw_mean", "hcand_mean", "z_virtual"):
        reference = train_data[feature].astype(np.float32)
        query = test_data[feature][queries].astype(np.float32)
        reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        neighbors = np.argsort(-(query @ reference.T), axis=1, kind="stable")[:, :count]
        fig, axes = plt.subplots(len(queries), count + 1, figsize=(2 * (count + 1), 2.2 * len(queries)), squeeze=False)
        selections[feature] = []
        for row, (index, matches) in enumerate(zip(queries, neighbors)):
            paths = [test_data["paths"][index]] + list(train_data["paths"][matches])
            labels = [test_data["labels"][index]] + list(train_data["labels"][matches])
            selections[feature].append({"query": str(paths[0]), "neighbors": [str(path) for path in paths[1:]]})
            for col, (path, label) in enumerate(zip(paths, labels)):
                with Image.open(os.path.join(root_dir, str(path))) as img:
                    axes[row, col].imshow(img.convert("RGB"))
                axes[row, col].set_title(f"{'Query' if col == 0 else 'NN ' + str(col)} | class {label}", fontsize=9)
                axes[row, col].axis("off")
        fig.suptitle(feature)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, feature + "_neighbors.png"), dpi=180)
        plt.close(fig)
    with open(os.path.join(output_dir, "neighbors.json"), "w") as handle:
        json.dump({"seed": seed, "selections": selections}, handle, indent=2)
