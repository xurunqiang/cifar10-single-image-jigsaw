"""
Unit tests for v3 dataset split and verification.
"""

import os
import tempfile
import json
import pytest
from v3.config import DataConfig
from v3.split import get_or_create_split, load_official_val_records, stratified_subsample_records


def test_stratified_subsample_covers_all_classes_deterministically():
    records = [
        {"class_idx": class_idx, "rel_path": f"{class_idx}/{sample_idx}"}
        for class_idx in range(4)
        for sample_idx in range(8)
    ]
    subset = stratified_subsample_records(records, max_samples=10, seed=123)
    counts = {}
    for record in subset:
        counts[record["class_idx"]] = counts.get(record["class_idx"], 0) + 1

    assert len(subset) == 10
    assert len(counts) == 4
    assert max(counts.values()) - min(counts.values()) <= 1
    assert subset == stratified_subsample_records(records, max_samples=10, seed=123)


def test_split_generation_and_counts():
    root_dir = DataConfig().root_dir
    if not os.path.exists(root_dir):
        pytest.skip("Tiny ImageNet data directory not found")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        split_path = tmp.name

    try:
        manifest = get_or_create_split(root_dir, split_path, seed=42)
        assert manifest["seed"] == 42
        assert len(manifest["wnids"]) == 200
        assert len(manifest["train"]) == 90000
        assert len(manifest["dev_val"]) == 10000

        # Check per-class count
        class_train_counts = {}
        for rec in manifest["train"]:
            c = rec["class_idx"]
            class_train_counts[c] = class_train_counts.get(c, 0) + 1

        assert len(class_train_counts) == 200
        for c, count in class_train_counts.items():
            assert count == 450, f"Class {c} has {count} train samples, expected 450"

        # Check official val records
        val_records = load_official_val_records(root_dir, manifest["wnid_to_idx"])
        assert len(val_records) == 10000

    finally:
        if os.path.exists(split_path):
            os.remove(split_path)
