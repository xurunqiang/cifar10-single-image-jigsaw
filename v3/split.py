"""
Stratified dataset split generator and verification for Tiny ImageNet:
- 200 classes, each 500 images in official train/
- Exactly 450 train / 50 dev-val per class with seed 42 (90,000 train / 10,000 dev-val)
- Preserves official val/ (10,000 images) as test/eval set
- Saves split manifest to JSON
"""

import os
import json
import hashlib
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional


def generate_tiny_imagenet_split(
    root_dir: str,
    output_split_file: str,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates and saves stratified 450/50 split of Tiny ImageNet train set.
    """
    train_dir = os.path.join(root_dir, "train")
    wnids_path = os.path.join(root_dir, "wnids.txt")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Read class wnids in order
    if os.path.exists(wnids_path):
        with open(wnids_path, "r") as f:
            wnids = [line.strip() for line in f if line.strip()]
    else:
        wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

    train_records: List[Dict[str, Any]] = []
    dev_val_records: List[Dict[str, Any]] = []

    for class_idx, wnid in enumerate(wnids):
        class_img_dir = os.path.join(train_dir, wnid, "images")
        if not os.path.exists(class_img_dir):
            class_img_dir = os.path.join(train_dir, wnid)

        all_imgs = sorted([f for f in os.listdir(class_img_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        assert len(all_imgs) == 500, f"Class {wnid} has {len(all_imgs)} images, expected 500"

        # Deterministic shuffle per class with seed + class_idx
        rng = random.Random(seed + class_idx * 10007)
        indices = list(range(len(all_imgs)))
        rng.shuffle(indices)

        train_indices = indices[:450]
        dev_indices = indices[450:]

        for i in train_indices:
            rel_path = os.path.relpath(os.path.join(class_img_dir, all_imgs[i]), root_dir)
            train_records.append({
                "rel_path": rel_path,
                "class_idx": class_idx,
                "wnid": wnid
            })

        for i in dev_indices:
            rel_path = os.path.relpath(os.path.join(class_img_dir, all_imgs[i]), root_dir)
            dev_val_records.append({
                "rel_path": rel_path,
                "class_idx": class_idx,
                "wnid": wnid
            })

    assert len(train_records) == 90000, f"Expected 90,000 train records, got {len(train_records)}"
    assert len(dev_val_records) == 10000, f"Expected 10,000 dev-val records, got {len(dev_val_records)}"

    manifest = {
        "seed": seed,
        "num_classes": len(wnids),
        "wnids": wnids,
        "wnid_to_idx": wnid_to_idx,
        "train": train_records,
        "dev_val": dev_val_records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_split_file)), exist_ok=True)
    with open(output_split_file, "w") as f:
        json.dump(manifest, f)

    print(f"Successfully generated stratified split: 90,000 train / 10,000 dev-val -> {output_split_file}")
    return manifest


def validate_manifest(manifest: Dict[str, Any], seed: int) -> None:
    wnids = manifest.get("wnids", [])
    mapping = {wnid: index for index, wnid in enumerate(wnids)}
    if len(wnids) != 200 or len(mapping) != 200 or manifest.get("seed") != seed or manifest.get("wnid_to_idx") != mapping:
        raise ValueError("Split class mapping/seed is invalid; choose a new split_file explicitly")
    seen = set()
    for name, expected in (("train", 450), ("dev_val", 50)):
        counts = Counter()
        for record in manifest.get(name, []):
            path, label, wnid = record["rel_path"], record["class_idx"], record["wnid"]
            parts = path.replace("\\", "/").split("/")
            if path in seen or len(parts) < 3 or parts[0] != "train" or ".." in parts or parts[1] != wnid or mapping.get(wnid) != label:
                raise ValueError("Split contains duplicate/overlapping paths or invalid class labels")
            seen.add(path)
            counts[label] += 1
        if counts != Counter({index: expected for index in range(200)}):
            raise ValueError(f"{name} must contain {expected} images per class")


def split_fingerprint(manifest: Dict[str, Any]) -> str:
    payload = {key: manifest[key] for key in ("seed", "wnids", "train", "dev_val")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get_or_create_split(root_dir: str, split_file: str, seed: int = 42) -> Dict[str, Any]:
    if os.path.exists(split_file) and os.path.getsize(split_file) > 0:
        with open(split_file) as handle:
            manifest = json.load(handle)
        validate_manifest(manifest, seed)
        with open(os.path.join(root_dir, "wnids.txt")) as handle:
            wnids = [line.strip() for line in handle if line.strip()]
        if manifest["wnids"] != wnids:
            raise ValueError("Dataset class order differs from the saved split")
        return manifest
    manifest = generate_tiny_imagenet_split(root_dir, split_file, seed=seed)
    validate_manifest(manifest, seed)
    return manifest


def load_official_val_records(root_dir: str, wnid_to_idx: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Loads official validation set (10,000 images) with ground truth labels.
    Used as the final test evaluation set.
    """
    val_dir = os.path.join(root_dir, "val")
    anno_path = os.path.join(val_dir, "val_annotations.txt")
    records = []

    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Official val annotations not found at: {anno_path}")

    with open(anno_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name, wnid = parts[0], parts[1]
                if wnid in wnid_to_idx:
                    rel_path = os.path.join("val", "images", img_name)
                    records.append({
                        "rel_path": rel_path,
                        "class_idx": wnid_to_idx[wnid],
                        "wnid": wnid
                    })

    assert len(records) == 10000, f"Expected 10,000 official val records, got {len(records)}"
    return records


def stratified_subsample_records(
    records: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Select a deterministic, near-balanced subset across the available classes."""
    if max_samples is None or max_samples >= len(records):
        return list(records)
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    by_class = defaultdict(list)
    for record in records:
        by_class[int(record["class_idx"])].append(record)

    rng = random.Random(seed)
    classes = sorted(by_class)
    rng.shuffle(classes)
    for class_idx in classes:
        rng.shuffle(by_class[class_idx])

    selected = []
    offsets = {class_idx: 0 for class_idx in classes}
    while len(selected) < max_samples:
        made_progress = False
        for class_idx in classes:
            offset = offsets[class_idx]
            if offset < len(by_class[class_idx]) and len(selected) < max_samples:
                selected.append(by_class[class_idx][offset])
                offsets[class_idx] = offset + 1
                made_progress = True
        if not made_progress:
            break

    return selected
