"""Training and reconstruction CLI. Full evaluation: python -m v3.evaluate."""
import argparse
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, AugmentationConfig, ModelConfig, TrainConfig
from .split import get_or_create_split, load_official_val_records, stratified_subsample_records, split_fingerprint
from .dataset import TinyImageNetDualDataset, TinyImageNetSingleDataset, collate_two_views, collate_single_view
from .model import JigsawSolverV3
from .trainer import JigsawTrainerV3
from .representation_eval import extract_representations, evaluate_knn_cosine
from .visualize import render_sample_cards, plot_training_curves


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    return device


def cmd_split(args):
    manifest = get_or_create_split(args.data_root, args.split_file, seed=args.seed)
    print(f"Verified {len(manifest['train'])} train / {len(manifest['dev_val'])} dev-val")


def cmd_train(args):
    device = resolve_device(args.device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint and checkpoint.get("format_version") != 2:
        raise ValueError("Old v3 architecture checkpoint: start a new run in a new save_dir")
    m_values = dict(checkpoint["model_config"]) if checkpoint else {}
    t_values = dict(checkpoint["train_config"]) if checkpoint else {}
    for key in ("grid_size", "content_dim", "local_layers", "local_heads", "global_layers", "global_heads", "global_ffn_dim", "cnn_stages"):
        value = getattr(args, key)
        if value is not None:
            m_values[key] = value
    for key in ("epochs", "batch_size", "lr", "seed", "num_workers", "feature_eval_interval", "feature_reference_samples", "feature_validation_samples"):
        value = getattr(args, key)
        if value is not None:
            t_values[key] = value
    if args.semantic_weight is not None:
        t_values.update(semantic_weight_max=args.semantic_weight, semantic_weight_start=args.semantic_weight / 10)
    t_values["device"] = str(device)
    if args.save_dir is not None:
        t_values["save_dir"] = args.save_dir
    m_cfg, t_cfg = ModelConfig(**m_values), TrainConfig(**t_values)
    meta = checkpoint.get("run_metadata", {}) if checkpoint else {}
    d_values = dict(meta.get("data_config", {}))
    for arg, key in (("data_root", "root_dir"), ("split_file", "split_file")):
        if getattr(args, arg) is not None:
            d_values[key] = getattr(args, arg)
    d_values["grid_size"] = m_cfg.grid_size
    d_cfg = DataConfig(**d_values)
    a_cfg = AugmentationConfig(**meta.get("augmentation_config", {}))
    if not args.resume and os.path.exists(os.path.join(t_cfg.save_dir, "latest.pt")):
        raise ValueError("save_dir already contains latest.pt; use --resume or a new --save_dir")
    random.seed(t_cfg.seed)
    np.random.seed(t_cfg.seed)
    torch.manual_seed(t_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t_cfg.seed)
    manifest = get_or_create_split(d_cfg.root_dir, d_cfg.split_file, seed=t_cfg.seed)
    run_metadata = {"data_config": asdict(d_cfg), "augmentation_config": asdict(a_cfg),
                    "split_fingerprint": split_fingerprint(manifest)}
    dataset_kwargs = dict(root_dir=d_cfg.root_dir, grid_size=d_cfg.grid_size,
                          seed_coord=d_cfg.seed_coord, aug_config=a_cfg)
    train_dataset = TinyImageNetDualDataset(records=manifest["train"], **dataset_kwargs)
    val_dataset = TinyImageNetSingleDataset(records=manifest["dev_val"], base_seed=t_cfg.seed, **dataset_kwargs)
    loader_kwargs = dict(batch_size=t_cfg.batch_size, num_workers=t_cfg.num_workers,
                         pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, collate_fn=collate_two_views,
                              generator=torch.Generator().manual_seed(t_cfg.seed), **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_single_view,
                            generator=torch.Generator().manual_seed(t_cfg.seed + 1), **loader_kwargs)
    model = JigsawSolverV3(m_cfg)
    trainer = JigsawTrainerV3(model, m_cfg, t_cfg, train_loader, val_loader, device, run_metadata)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    feature_loaders = []
    if t_cfg.feature_eval_interval:
        for name, cap, offset in (("train", t_cfg.feature_reference_samples, 3407),
                                  ("dev_val", t_cfg.feature_validation_samples, 9182)):
            records = stratified_subsample_records(manifest[name], cap, offset)
            dataset = TinyImageNetSingleDataset(records=records, base_seed=t_cfg.seed, **dataset_kwargs)
            feature_loaders.append(DataLoader(dataset, shuffle=False, collate_fn=collate_single_view,
                generator=torch.Generator().manual_seed(offset), **loader_kwargs))
    print(f"v3 grid={m_cfg.grid_size}, center={d_cfg.seed_coord}, BFS, device={device}")
    for epoch in range(trainer.start_epoch, t_cfg.epochs + 1):
        try:
            tr_stats = trainer.train_epoch(epoch)
            val_stats = trainer.validate()
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Restart in a new save_dir with a smaller --batch_size; configuration was not changed automatically.")
            raise
        feature_acc = None
        is_best_feat = False
        if feature_loaders and (epoch % t_cfg.feature_eval_interval == 0 or epoch == t_cfg.epochs):
            reference = extract_representations(model, feature_loaders[0], device)
            query = extract_representations(model, feature_loaders[1], device)
            feature_acc = evaluate_knn_cosine(reference["z_virtual"], reference["labels"],
                query["z_virtual"], query["labels"], device=str(device))
            is_best_feat = feature_acc > trainer.best_feat_acc
            if is_best_feat:
                trainer.best_feat_acc = feature_acc
            del reference, query
        is_best = (val_stats["val_patch_acc"], val_stats["val_perfect_acc"]) > (trainer.best_acc, trainer.best_perfect_acc)
        if is_best:
            trainer.best_acc, trainer.best_perfect_acc = val_stats["val_patch_acc"], val_stats["val_perfect_acc"]
        trainer.history.append({**tr_stats, **val_stats, "feature_knn_acc": feature_acc})
        if is_best:
            trainer.save_checkpoint("best_jigsaw.pt")
        if is_best_feat:
            trainer.save_checkpoint("best_feat.pt")
        trainer.save_checkpoint("latest.pt")
        history_path = os.path.join(t_cfg.save_dir, "history.json")
        with open(history_path + ".tmp", "w") as handle:
            json.dump(trainer.history, handle, indent=2)
        os.replace(history_path + ".tmp", history_path)
        plot_training_curves(trainer.history, os.path.join(t_cfg.save_dir, "training_curves.png"))
        print(f"Epoch {epoch:03d}/{t_cfg.epochs} | Loss {tr_stats['loss_total']:.4f} | "
              f"Jig/VIC {tr_stats['loss_jigsaw']:.4f}/{tr_stats['loss_vicreg']:.4f} | "
              f"TF {tr_stats['tf_prob']:.2f} | Actual/Conflict {tr_stats['actual_ratio']:.2f}/{tr_stats['conflicts_ratio']:.2f} | "
              f"Val Patch/Perfect {val_stats['val_patch_acc']:.2f}%/{val_stats['val_perfect_acc']:.2f}% | "
              f"Dev feature 5-NN {feature_acc if feature_acc is not None else 'not measured'}")


def cmd_visualize(args):
    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("format_version") != 2:
        raise ValueError("Checkpoint uses an older v3 architecture")
    m_cfg = ModelConfig(**ckpt["model_config"])
    model = JigsawSolverV3(m_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seed = ckpt["train_config"]["seed"]
    manifest = get_or_create_split(args.data_root, args.split_file, seed=seed)
    if split_fingerprint(manifest) != ckpt["run_metadata"]["split_fingerprint"]:
        raise ValueError("Dataset split differs from checkpoint")
    records = load_official_val_records(args.data_root, manifest["wnid_to_idx"])
    dataset = TinyImageNetSingleDataset(args.data_root, records, grid_size=m_cfg.grid_size,
        base_seed=seed, aug_config=AugmentationConfig(**ckpt["run_metadata"]["augmentation_config"]))
    render_sample_cards(model, dataset, device, args.output_dir, args.num_cards,
                        failure_only=args.failures, selection_seed=args.selection_seed)


def main():
    parser = argparse.ArgumentParser(description="v3 Tiny ImageNet")
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = DataConfig()
    split = sub.add_parser("split")
    split.add_argument("--data_root", default=defaults.root_dir)
    split.add_argument("--split_file", default=defaults.split_file)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=cmd_split)
    train = sub.add_parser("train")
    for name in ("data_root", "split_file", "save_dir", "resume"):
        train.add_argument("--" + name)
    for name in ("epochs", "batch_size", "seed", "num_workers", "grid_size", "content_dim",
                 "local_layers", "local_heads", "global_layers", "global_heads", "global_ffn_dim",
                 "feature_eval_interval", "feature_reference_samples", "feature_validation_samples"):
        train.add_argument("--" + name, type=int)
    train.add_argument("--cnn_stages", type=int, nargs=3)
    train.add_argument("--lr", type=float)
    train.add_argument("--semantic_weight", type=float, help="0 disables semantic training; default max weight is 0.1")
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=cmd_train)
    vis = sub.add_parser("visualize")
    vis.add_argument("--checkpoint", required=True)
    vis.add_argument("--data_root", default=defaults.root_dir)
    vis.add_argument("--split_file", default=defaults.split_file)
    vis.add_argument("--output_dir", default="visualizations/v3")
    vis.add_argument("--num_cards", type=int, default=10)
    vis.add_argument("--device", default="cuda")
    vis.add_argument("--failures", action="store_true")
    vis.add_argument("--selection_seed", type=int, default=3407)
    vis.set_defaults(func=cmd_visualize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
