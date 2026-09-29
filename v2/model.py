"""
v2 Neural Architecture for Jigsaw Puzzle Reconstruction:
1. ContentEncoder: Small CNN with 2x2 spatial pooling and 96-dim output
2. LocalBranch: 1-layer, 4-head cross-attention over 4 directional neighbors (U, R, D, L)
3. GlobalBranch: 2-layer, 4-head Transformer over all K candidates
4. FuseMLP & Matching: MLP query generation and dot-product matching
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ContentEncoder(nn.Module):
    """
    Small CNN encoder for patch images: (B, 3, P, P) -> (B, 96).
    """
    def __init__(self, content_dim: int = 96):
        super().__init__()
        self.content_dim = content_dim

        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(6, 48)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(48, content_dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(12, content_dim)
        self.act2 = nn.GELU()

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Linear(content_dim * 2 * 2, content_dim)
        self.norm = nn.LayerNorm(content_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3, P, P)
        returns: (N, content_dim)
        """
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.act2(self.gn2(self.conv2(h)))
        h = self.pool(h)
        h_flat = h.view(h.size(0), -1)
        feat = self.norm(self.proj(h_flat))
        return feat


class LocalBranch(nn.Module):
    """
    Local semantic branch:
    Reads 4 placed neighbors (Top, Right, Bottom, Left) with relative direction embeddings.
    Uses 1-layer, 4-head cross-attention; missing/unplaced neighbors are masked out.
    """
    def __init__(self, content_dim: int = 96, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.content_dim = content_dim
        self.num_heads = num_heads

        # 4 directional embeddings: 0: Top, 1: Right, 2: Bottom, 3: Left
        self.dir_embed = nn.Parameter(torch.randn(4, content_dim) * 0.02)
        self.register_buffer("neighbor_offsets", torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]]), persistent=False)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(content_dim)
        self.norm2 = nn.LayerNorm(content_dim)

        self.mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(content_dim * mlp_ratio, content_dim),
        )

    def forward(
        self,
        query_pos: torch.Tensor,              # (B, 1, D) target slot pos embed
        raw_feats: torch.Tensor,              # (B, K, D) candidate features
        grid_placed: torch.Tensor,            # (B, G, G) candidate index placed or -1
        target_coords: torch.Tensor,          # (B, 2) (r, c)
        grid_size: int
    ) -> torch.Tensor:
        """
        Extracts up to 4 neighbors for each sample, masks missing/unplaced neighbors,
        and applies 1-layer cross-attention.
        Returns: (B, D) local context vector.
        """
        B, K, D = raw_feats.shape
        device = raw_feats.device

        # Gather all four directions in one operation, without per-direction GPU synchronizations.
        coords = target_coords[:, None, :] + self.neighbor_offsets[None, :, :]
        valid = ((coords >= 0) & (coords < grid_size)).all(dim=-1)
        safe_coords = coords.clamp(0, grid_size - 1)
        batch_idx = torch.arange(B, device=device)[:, None]
        placed_cand = grid_placed[batch_idx, safe_coords[..., 0], safe_coords[..., 1]]
        valid = valid & (placed_cand >= 0)
        neighbor_feats = raw_feats[batch_idx, placed_cand.clamp(0, K - 1)] + self.dir_embed
        neighbor_feats = neighbor_feats.masked_fill(~valid[..., None], 0.0)
        key_padding_mask = ~valid
        all_masked = key_padding_mask.all(dim=1)
        # Keep a zero key for empty neighborhoods; discard that query's output below.
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_masked

        # Cross Attention: Q=(B, 1, D), K/V=(B, 4, D)
        attn_out, _ = self.cross_attn(
            query=query_pos,
            key=neighbor_feats,
            value=neighbor_feats,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, 1, D)

        # Residual & Norm + MLP
        h = self.norm1(query_pos + attn_out)
        h = self.norm2(h + self.mlp(h))

        # If all were masked, zero out the context
        h = h.masked_fill(all_masked[:, None, None], 0.0)

        return h.squeeze(1)  # (B, D)


class GlobalBranch(nn.Module):
    """
    Global semantic branch:
    Reads all K candidates at each step using a 2-layer, 4-head Transformer.
    - Placed candidates receive their placement grid position embedding.
    - Unplaced candidates receive an unplaced state embedding (no true position).
    - Candidate sequence does NOT include candidate index position embedding.
    - Target empty slot reads global features via cross-attention.
    """
    def __init__(self, content_dim: int = 96, num_heads: int = 4, num_layers: int = 2, mlp_ratio: int = 2):
        super().__init__()
        self.content_dim = content_dim

        # Unplaced state embedding
        self.unplaced_embed = nn.Parameter(torch.randn(1, 1, content_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=content_dim,
            nhead=num_heads,
            dim_feedforward=content_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Cross attention for target empty slot to query global candidates
        self.target_cross_attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(content_dim)
        self.norm2 = nn.LayerNorm(content_dim)
        self.mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(content_dim * mlp_ratio, content_dim),
        )

    def forward(
        self,
        query_pos: torch.Tensor,              # (B, 1, D) target slot pos embed
        raw_feats: torch.Tensor,              # (B, K, D)
        cand_to_slot: torch.Tensor,           # (B, K, 2) coords or (-1, -1)
        used_candidates: torch.Tensor,        # (B, K) bool
        pos_embed_table: nn.Parameter         # (max_grid, max_grid, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            F_global: (B, D) global context queried by target slot
            H_cand: (B, K, D) globally enhanced candidate representations
        """
        B, K, D = raw_feats.shape
        device = raw_feats.device

        # Build position/state embeddings for all K candidates
        # cand_to_slot: (B, K, 2)
        safe_r = cand_to_slot[:, :, 0].clamp(0, pos_embed_table.shape[0] - 1)
        safe_c = cand_to_slot[:, :, 1].clamp(0, pos_embed_table.shape[1] - 1)
        placed_pos_embeds = pos_embed_table[safe_r, safe_c]  # (B, K, D)

        unplaced_embeds = self.unplaced_embed.expand(B, K, D)

        placed_mask = used_candidates.unsqueeze(-1)  # (B, K, 1)
        state_embeds = torch.where(placed_mask, placed_pos_embeds, unplaced_embeds)  # (B, K, D)

        # Input to Transformer: raw_feats + state_embeds (NO candidate index encoding)
        transformer_input = raw_feats + state_embeds  # (B, K, D)

        H_cand = self.transformer(transformer_input)  # (B, K, D)

        # Target slot queries H_cand
        attn_out, _ = self.target_cross_attn(
            query=query_pos,
            key=H_cand,
            value=H_cand,
            need_weights=False,
        )  # (B, 1, D)

        h = self.norm1(query_pos + attn_out)
        h = self.norm2(h + self.mlp(h))
        F_global = h.squeeze(1)  # (B, D)

        return F_global, H_cand


class JigsawSolverV2(nn.Module):
    """
    Unified Jigsaw Puzzle Reconstruction Model v2:
    - ContentEncoder for candidates (encoded once)
    - Local cross-attention branch
    - Global Transformer branch
    - Fusion MLP and dot-product matching
    - Supports modes: "both", "local_only", "global_only"
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.content_dim = config.content_dim
        self.mode = config.mode
        self.temperature = config.temperature

        # 2D Grid Position Embeddings (table size 16x16 to easily accommodate up to 7x7)
        max_grid = 16
        self.pos_embed = nn.Parameter(torch.randn(max_grid, max_grid, self.content_dim) * 0.02)

        # CNN Content Encoder
        self.encoder = ContentEncoder(content_dim=self.content_dim)

        # Semantic branches
        self.local_branch = LocalBranch(
            content_dim=self.content_dim,
            num_heads=config.local_heads,
            mlp_ratio=config.mlp_ratio
        )
        self.global_branch = GlobalBranch(
            content_dim=self.content_dim,
            num_heads=config.global_heads,
            num_layers=config.global_layers,
            mlp_ratio=config.mlp_ratio
        )

        # Fusion MLP
        # Input dimension:
        # "both": [F_local, F_global, target_pos] -> 3 * D
        # "local_only": [F_local, target_pos] -> 2 * D
        # "global_only": [F_global, target_pos] -> 2 * D
        if self.mode == "both":
            in_dim = self.content_dim * 3
        elif self.mode in ["local_only", "global_only"]:
            in_dim = self.content_dim * 2
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.fuse_mlp = nn.Sequential(
            nn.Linear(in_dim, self.content_dim),
            nn.GELU(),
            nn.Linear(self.content_dim, self.content_dim),
            nn.LayerNorm(self.content_dim)
        )

        # Projection for local_only mode candidate features
        if self.mode == "local_only":
            self.cand_proj = nn.Sequential(
                nn.Linear(self.content_dim, self.content_dim),
                nn.LayerNorm(self.content_dim)
            )

    def encode_candidates(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode candidate patches once.
        patches: (B, K, 3, P, P)
        returns: (B, K, D)
        """
        B, K, C, P, _ = patches.shape
        flat_patches = patches.reshape(B * K, C, P, P)
        raw_feats = self.encoder(flat_patches)  # (B*K, D)
        return raw_feats.view(B, K, self.content_dim)

    def score_step(
        self,
        raw_feats: torch.Tensor,              # (B, K, D)
        grid_placed: torch.Tensor,            # (B, G, G) candidate index or -1
        used_candidates: torch.Tensor,        # (B, K) bool
        cand_to_slot: torch.Tensor,           # (B, K, 2) coords or -1
        target_coords: torch.Tensor,          # (B, 2) (r, c)
        grid_size: int
    ) -> torch.Tensor:
        """
        Score all K candidates for the target empty slot.
        Does NOT receive ground truth permutation.
        Returns: logits of shape (B, K)
        """
        B, K, D = raw_feats.shape
        device = raw_feats.device

        # Target slot position embedding: (B, 1, D)
        target_r = target_coords[:, 0].clamp(0, self.pos_embed.shape[0] - 1)
        target_c = target_coords[:, 1].clamp(0, self.pos_embed.shape[1] - 1)
        query_pos = self.pos_embed[target_r, target_c].unsqueeze(1)  # (B, 1, D)
        target_pos_flat = query_pos.squeeze(1)  # (B, D)

        if self.mode == "both":
            F_local = self.local_branch(
                query_pos=query_pos,
                raw_feats=raw_feats,
                grid_placed=grid_placed,
                target_coords=target_coords,
                grid_size=grid_size
            )
            F_global, H_cand = self.global_branch(
                query_pos=query_pos,
                raw_feats=raw_feats,
                cand_to_slot=cand_to_slot,
                used_candidates=used_candidates,
                pos_embed_table=self.pos_embed
            )
            fused = torch.cat([F_local, F_global, target_pos_flat], dim=-1)
            Q_match = self.fuse_mlp(fused)  # (B, D)
            cand_features = H_cand  # (B, K, D)

        elif self.mode == "local_only":
            F_local = self.local_branch(
                query_pos=query_pos,
                raw_feats=raw_feats,
                grid_placed=grid_placed,
                target_coords=target_coords,
                grid_size=grid_size
            )
            fused = torch.cat([F_local, target_pos_flat], dim=-1)
            Q_match = self.fuse_mlp(fused)  # (B, D)
            cand_features = self.cand_proj(raw_feats)  # (B, K, D)

        elif self.mode == "global_only":
            F_global, H_cand = self.global_branch(
                query_pos=query_pos,
                raw_feats=raw_feats,
                cand_to_slot=cand_to_slot,
                used_candidates=used_candidates,
                pos_embed_table=self.pos_embed
            )
            fused = torch.cat([F_global, target_pos_flat], dim=-1)
            Q_match = self.fuse_mlp(fused)  # (B, D)
            cand_features = H_cand  # (B, K, D)

        # Cosine similarity matching scaled by temperature: (B, K, D) @ (B, D, 1) -> (B, K)
        query_norm = F.normalize(Q_match, p=2, dim=-1)
        cand_norm = F.normalize(cand_features, p=2, dim=-1)
        scores = torch.bmm(cand_norm, query_norm.unsqueeze(-1)).squeeze(-1) / self.temperature

        return scores  # (B, K)
