"""
Downstream Representation Evaluation Suite for v3:
- Extracts three representation types:
  1. raw_mean: mean over 25 patch CNN encoder features (256-dim)
  2. hcand_mean: mean over 25 globally contextualized H_cand features (256-dim)
  3. z_virtual: coordinate-free virtual patch embedding (256-dim)
- Evaluation Metrics:
  1. 5-NN Cosine Classifier Accuracy (Tiny ImageNet 200 classes)
  2. K-Means (200 clusters):
     - Cluster matching accuracy (Hungarian maximum weight bipartite matching)
     - Adjusted Rand Index (ARI)
     - Normalized Mutual Information (NMI)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from .solver import solve_batch


@torch.no_grad()
def extract_representations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Extracts raw_mean, hcand_mean, and z_virtual representations across dataloader.
    """
    model.eval()
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    raw_means = []
    hcand_means = []
    z_virtuals = []
    all_labels = []

    all_paths = []
    samples_collected = 0
    for batch in dataloader:
        candidates = batch["candidates"].to(device)
        seed_cands = batch["seed_cand"].to(device)
        seed_coords = batch["seed_coord"].to(device)
        labels = batch["class_idx"]

        if max_samples is not None:
            remaining = max_samples - samples_collected
            if remaining <= 0:
                break
            candidates = candidates[:remaining]
            seed_cands = seed_cands[:remaining]
            seed_coords = seed_coords[:remaining]
            labels = labels[:remaining]

        B, K = candidates.shape[:2]

        # Autonomous puzzle solving
        out = solve_batch(
            model=model,
            patches=candidates,
            seed_cands=seed_cands,
            seed_coords=seed_coords,
            grid_size=model.config.grid_size
        )

        raw_feats = out["raw_feats"]          # (B, K, D)
        z_virt = out["z_virtual"]              # (B, D)

        # Contextualized H_final from placed board
        safe_r = out["cand_to_slot"][:, :, 0].clamp(0, model.pos_embed.shape[0] - 1)
        safe_c = out["cand_to_slot"][:, :, 1].clamp(0, model.pos_embed.shape[1] - 1)
        placed_pos = model.pos_embed[safe_r, safe_c]
        h_cand_final = model.global_branch.run_transformer(raw_feats + placed_pos)  # (B, K, D)

        raw_mean = raw_feats.mean(dim=1).cpu().numpy()
        hcand_mean = h_cand_final.mean(dim=1).cpu().numpy()
        z_virt_np = z_virt.cpu().numpy()
        labels_np = labels.numpy()

        raw_means.append(raw_mean)
        hcand_means.append(hcand_mean)
        z_virtuals.append(z_virt_np)
        all_labels.append(labels_np)
        all_paths.extend(batch.get("rel_paths", [""] * B)[:B])

        samples_collected += B
        if max_samples is not None and samples_collected >= max_samples:
            break

    if not raw_means:
        raise ValueError("Representation dataset is empty")
    return {
        "paths": np.asarray(all_paths),
        "raw_mean": np.concatenate(raw_means, axis=0)[:max_samples],
        "hcand_mean": np.concatenate(hcand_means, axis=0)[:max_samples],
        "z_virtual": np.concatenate(z_virtuals, axis=0)[:max_samples],
        "labels": np.concatenate(all_labels, axis=0)[:max_samples]
    }


def evaluate_knn_cosine(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5,
    batch_size: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Computes k-NN classification accuracy using cosine similarity.
    Batched matrix multiplication on GPU/CPU for efficiency.
    """
    if len(train_feats) < k or len(test_feats) == 0 or k < 1:
        raise ValueError("k-NN needs at least k reference images and a nonempty query set")
    t_train = torch.from_numpy(train_feats).float().to(device)
    t_test = torch.from_numpy(test_feats).float().to(device)
    l_train = torch.from_numpy(train_labels).long().to(device)
    l_test = torch.from_numpy(test_labels).long().to(device)

    # Normalize vectors
    t_train = F.normalize(t_train, p=2, dim=-1)
    t_test = F.normalize(t_test, p=2, dim=-1)

    num_test = t_test.shape[0]
    correct = 0

    for i in range(0, num_test, batch_size):
        batch_test = t_test[i:i + batch_size]              # (b, D)
        batch_labels = l_test[i:i + batch_size]            # (b,)

        sims = torch.matmul(batch_test, t_train.T)         # (b, N_train)
        _, topk_indices = torch.topk(sims, k=k, dim=-1)    # (b, k)
        topk_labels = l_train[topk_indices]                # (b, k)

        # Mode / majority vote
        preds, _ = torch.mode(topk_labels, dim=-1)
        correct += (preds == batch_labels).sum().item()

    return float(correct) / float(num_test)


def evaluate_kmeans_clustering(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    num_clusters: int = 200,
    seed: int = 42
) -> Dict[str, float]:
    """Fit centers and cluster-to-class mapping on reference data; freeze both for evaluation.

    Labels never enter KMeans.fit. Reference labels only align cluster IDs after fitting.
    Evaluation labels are used exclusively for scoring the fixed predictions.
    """
    if len(train_feats) < num_clusters or len(test_feats) == 0:
        raise ValueError("K-means requires at least num_clusters reference samples and nonempty evaluation data")
    normalize = lambda x: x.astype(np.float32) / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    train_feats, test_feats = normalize(train_feats), normalize(test_feats)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    train_clusters = kmeans.fit_predict(train_feats)
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    if np.any((train_labels < 0) | (train_labels >= num_clusters)):
        raise ValueError("Reference labels must be in [0, num_clusters)")
    np.add.at(contingency, (train_clusters, train_labels), 1)
    cluster_ids, class_ids = linear_sum_assignment(-contingency)
    mapping = np.full(num_clusters, -1, dtype=np.int64)
    mapping[cluster_ids] = class_ids
    test_clusters = kmeans.predict(test_feats)
    return {
        "cluster_acc": float(np.mean(mapping[test_clusters] == test_labels)),
        "ari": float(adjusted_rand_score(test_labels, test_clusters)),
        "nmi": float(normalized_mutual_info_score(test_labels, test_clusters))
    }


def evaluate_all_representations(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    k: int = 5,
    num_clusters: int = 200,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates 5-NN and K-Means for all three representation types:
    raw_mean, hcand_mean, z_virtual.
    """
    results: Dict[str, Dict[str, float]] = {}
    feature_types = ["raw_mean", "hcand_mean", "z_virtual"]

    for feat_name in feature_types:
        tr_feats = train_data[feat_name]
        te_feats = test_data[feat_name]
        tr_y = train_data["labels"]
        te_y = test_data["labels"]

        knn_acc = evaluate_knn_cosine(tr_feats, tr_y, te_feats, te_y, k=k, device=device)
        km_res = evaluate_kmeans_clustering(tr_feats, tr_y, te_feats, te_y, num_clusters=num_clusters, seed=seed)

        results[feat_name] = {
            "knn_5_acc": knn_acc,
            "kmeans_cluster_acc": km_res["cluster_acc"],
            "kmeans_ari": km_res["ari"],
            "kmeans_nmi": km_res["nmi"]
        }

    return results
