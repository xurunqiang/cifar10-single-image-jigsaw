"""
Full Evaluation Runner for v3:
- Autonomous Jigsaw Puzzle solving evaluation (Patch Acc, Perfect Acc, BFS Step Accuracies)
- Representation Evaluation (5-NN, K-Means Cluster Acc, ARI, NMI on raw_mean, hcand_mean, z_virtual)
- Summary report and JSON export
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig, TrainConfig, AugmentationConfig
from .model import JigsawSolverV3
from .dataset import TinyImageNetSingleDataset, collate_single_view, generate_bfs_order
from .split import get_or_create_split, load_official_val_records, stratified_subsample_records, split_fingerprint
from .solver import solve_batch, compute_puzzle_accuracy, compute_neighbor_accuracy
from .representation_eval import extract_representations, evaluate_all_representations
from .visualize import plot_nearest_neighbors


@torch.no_grad()
def evaluate_jigsaw_detailed(
    model: JigsawSolverV3,
    dataloader: DataLoader,
    device: torch.device,
    grid_size: int = 5,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detailed evaluation of autonomous jigsaw solving on a dataset.
    Computes overall patch accuracy, perfect accuracy, and step-wise accuracy.
    """
    model.eval()
    bfs_order = generate_bfs_order(grid_size, (grid_size // 2, grid_size // 2))
    num_steps = len(bfs_order)

    total_samples = 0
    total_patch_acc = 0.0
    total_perfect = 0.0
    total_neighbors = 0.0
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    step_correct_counts = np.zeros(num_steps, dtype=np.int64)

    for batch in dataloader:
        candidates = batch["candidates"].to(device)
        seed_cands = batch["seed_cand"].to(device)
        seed_coords = batch["seed_coord"].to(device)
        target_mappings = batch["target_mapping"].to(device)
        if max_samples is not None:
            remaining = max_samples - total_samples
            if remaining <= 0:
                break
            candidates, seed_cands = candidates[:remaining], seed_cands[:remaining]
            seed_coords, target_mappings = seed_coords[:remaining], target_mappings[:remaining]
        B = candidates.shape[0]

        out = solve_batch(
            model=model,
            patches=candidates,
            seed_cands=seed_cands,
            seed_coords=seed_coords,
            grid_size=grid_size,
            expansion_order=bfs_order
        )

        grid_placed = out["grid_placed"]
        patch_acc, perfect = compute_puzzle_accuracy(
            grid_placed=grid_placed,
            target_mapping=target_mappings,
            seed_coords=seed_coords,
            grid_size=grid_size
        )

        total_samples += B
        total_patch_acc += float(patch_acc.sum().item())
        total_perfect += float(perfect.sum().item())
        total_neighbors += float(compute_neighbor_accuracy(grid_placed, target_mappings).sum().item())

        # Step-wise accuracy
        batch_idx = torch.arange(B, device=device)
        for step_idx, (r, c) in enumerate(bfs_order):
            step_matches = (grid_placed[:, r, c] == target_mappings[:, r, c])
            step_correct_counts[step_idx] += int(step_matches.sum().item())

        if max_samples is not None and total_samples >= max_samples:
            break

    mean_patch_acc = (total_patch_acc / max(1, total_samples)) * 100.0
    mean_perfect_acc = (total_perfect / max(1, total_samples)) * 100.0
    step_accuracies = [(count / max(1, total_samples)) * 100.0 for count in step_correct_counts]

    return {
        "total_samples": total_samples,
        "patch_accuracy": mean_patch_acc,
        "perfect_accuracy": mean_perfect_acc,
        "neighbor_accuracy": 100 * total_neighbors / max(1, total_samples),
        "step_accuracies": step_accuracies,
        "bfs_order": bfs_order
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 Jigsaw Solver & Representations")
    defaults = DataConfig()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default=defaults.root_dir)
    parser.add_argument("--split_file", type=str, default=defaults.split_file)
    parser.add_argument("--eval_set", type=str, choices=["official_val", "dev_val"], default="official_val")
    parser.add_argument("--eval_repr", action="store_true", help="Run representation evaluation (5-NN, K-Means)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap evaluation samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default=None, help="Export metrics to JSON")
    parser.add_argument("--reference_samples", type=int, default=10000)
    parser.add_argument("--representation_samples", type=int, default=10000)
    parser.add_argument("--features_npz", help="Export train/test feature arrays and image paths")
    parser.add_argument("--visualization_dir", help="Output raw/Hcand/virtual nearest-neighbor images")
    parser.add_argument("--random_init", action="store_true", help="Same architecture with random weights, for the untrained baseline")
    parser.add_argument("--random_seed", type=int, default=3407)
    args = parser.parse_args()
    args.eval_repr = args.eval_repr or bool(args.features_npz or args.visualization_dir)
    if args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("batch_size and max_samples must be positive")
    if args.eval_repr:
        reference_cap = args.reference_samples if args.max_samples is None else args.max_samples
        query_cap = args.representation_samples if args.max_samples is None else args.max_samples
        if reference_cap < 200 or query_cap < 200:
            parser.error("200-class representation evaluation requires at least 200 reference and query samples")

    device = torch.device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint} on {device}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if ckpt.get("format_version") != 2:
        raise ValueError("Old v3 architecture checkpoint: evaluation requires the corrected architecture")
    seed = ckpt["train_config"]["seed"]
    aug = AugmentationConfig(**ckpt["run_metadata"]["augmentation_config"])
    if args.random_init:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
    # Reconstruct configs
    m_cfg = ModelConfig(**ckpt["model_config"])
    model = JigsawSolverV3(m_cfg).to(device)
    if not args.random_init:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load records
    manifest = get_or_create_split(args.data_root, args.split_file, seed=seed)
    if split_fingerprint(manifest) != ckpt["run_metadata"]["split_fingerprint"]:
        raise ValueError("Evaluation split differs from checkpoint")
    if args.eval_set == "official_val":
        records = load_official_val_records(args.data_root, manifest["wnid_to_idx"])
    else:
        records = manifest["dev_val"]

    puzzle_records = stratified_subsample_records(
        records, args.max_samples, seed=3407
    ) if args.max_samples is not None else records

    dataset = TinyImageNetSingleDataset(
        root_dir=args.data_root,
        records=puzzle_records,
        grid_size=m_cfg.grid_size,
        seed_coord=(m_cfg.grid_size // 2, m_cfg.grid_size // 2), base_seed=seed, aug_config=aug
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_single_view
    )

    print(f"\n--- 1. Evaluating Jigsaw Reconstruction ({len(dataset)} images) ---")
    jigsaw_metrics = evaluate_jigsaw_detailed(model, dataloader, device, grid_size=m_cfg.grid_size, max_samples=args.max_samples)
    print(f"Evaluated Samples: {jigsaw_metrics['total_samples']}")
    print(f"Non-seed Patch Accuracy: {jigsaw_metrics['patch_accuracy']:.2f}%")
    print(f"Perfect Reconstruction Accuracy: {jigsaw_metrics['perfect_accuracy']:.2f}%")

    results = {"jigsaw": jigsaw_metrics, "checkpoint": args.checkpoint, "eval_set": args.eval_set,
               "random_init": args.random_init, "random_seed": args.random_seed,
               "split_fingerprint": split_fingerprint(manifest),
               "kmeans_protocol": "L2 normalized; centers fitted on training references; training-label Hungarian mapping frozen before evaluation"}

    if args.eval_repr:
        print("\n--- 2. Extracting Representations for Downstream Evaluation ---")
        # For KNN, we extract features from dev-val as reference / test or train / test
        cap_train = min(args.reference_samples, len(manifest["train"])) if args.max_samples is None else min(args.max_samples, len(manifest["train"]))
        cap_test = min(args.representation_samples, len(records)) if args.max_samples is None else min(args.max_samples, len(records))
        repr_train_records = stratified_subsample_records(manifest["train"], cap_train, seed=3407)
        repr_test_records = stratified_subsample_records(records, cap_test, seed=9182)

        train_dataset = TinyImageNetSingleDataset(
            root_dir=args.data_root,
            records=repr_train_records,
            grid_size=m_cfg.grid_size,
            seed_coord=(m_cfg.grid_size // 2, m_cfg.grid_size // 2), base_seed=seed, aug_config=aug
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_single_view
        )

        repr_test_dataset = TinyImageNetSingleDataset(
            root_dir=args.data_root,
            records=repr_test_records,
            grid_size=m_cfg.grid_size,
            seed_coord=(m_cfg.grid_size // 2, m_cfg.grid_size // 2), base_seed=seed, aug_config=aug
        )
        repr_test_loader = DataLoader(
            repr_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_single_view
        )

        print(f"Extracting train representations (capped at {cap_train})...")
        train_data = extract_representations(model, train_loader, device, max_samples=cap_train)
        print(f"Extracting test representations (capped at {cap_test})...")
        test_data = extract_representations(model, repr_test_loader, device, max_samples=cap_test)

        print("Evaluating 5-NN and K-Means ARI / NMI...")
        repr_metrics = evaluate_all_representations(train_data, test_data, k=5, num_clusters=200, device=str(device))
        results["representations"] = repr_metrics
        results["representation_samples"] = {"train": len(train_data["labels"]), "evaluation": len(test_data["labels"])}
        if args.features_npz:
            os.makedirs(os.path.dirname(os.path.abspath(args.features_npz)), exist_ok=True)
            arrays = {"train_" + key: value for key, value in train_data.items()}
            arrays.update({"test_" + key: value for key, value in test_data.items()})
            np.savez_compressed(args.features_npz, **arrays)
        if args.visualization_dir:
            plot_nearest_neighbors(train_data, test_data, args.data_root, args.visualization_dir)

        print("\nRepresentation Results Summary:")
        for feat_name, scores in repr_metrics.items():
            print(f"  [{feat_name.upper()}]:")
            print(f"    5-NN Cosine Accuracy:   {scores['knn_5_acc'] * 100:.2f}%")
            print(f"    K-Means Fixed-map Acc: {scores['kmeans_cluster_acc'] * 100:.2f}%")
            print(f"    K-Means ARI:           {scores['kmeans_ari']:.4f}")
            print(f"    K-Means NMI:           {scores['kmeans_nmi']:.4f}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved evaluation metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
