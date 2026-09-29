"""
Neural Architecture for v3: Jigsaw Puzzle & Whole-Image Representation Learning:
1. PatchCNNEncoder: 3-stage ResNet (13x13 -> 7x7 -> 4x4 -> 2x2 pool -> 256)
2. LocalBranch: 2-layer, 8-head cross-attention over 4 directional neighbors
3. GlobalBranch: 4-layer, 8-head Pre-LN Transformer with gradient checkpointing
4. FuseMLP & Matching: Fuses local context, global context, and target pos embedding
5. VirtualPatchModule: Learnable coordinate-free query attending to final placed board -> Z_virtual
"""

from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .cnn_encoder import PatchCNNEncoder


class LocalBranch(nn.Module):
    """
    Local branch:
    Reads up to 4 placed neighbors (Top, Right, Bottom, Left) with relative direction embeddings.
    2-layer, 8-head cross-attention with Pre-LN.
    """
    def __init__(self, content_dim: int = 256, num_heads: int = 8, num_layers: int = 2, mlp_ratio: int = 2):
        super().__init__()
        self.content_dim = content_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 4 directional embeddings: 0: Top, 1: Right, 2: Bottom, 3: Left
        self.dir_embed = nn.Parameter(torch.randn(4, content_dim) * 0.02)
        self.register_buffer("neighbor_offsets", torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]]), persistent=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "norm_q": nn.LayerNorm(content_dim),
                "norm_kv": nn.LayerNorm(content_dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=content_dim,
                    num_heads=num_heads,
                    batch_first=True
                ),
                "norm_mlp": nn.LayerNorm(content_dim),
                "mlp": nn.Sequential(
                    nn.Linear(content_dim, content_dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(content_dim * mlp_ratio, content_dim)
                )
            })
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(content_dim)

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
        and applies 2-layer Pre-LN cross-attention.
        Returns: (B, D) local context vector.
        """
        B, K, D = raw_feats.shape
        device = raw_feats.device

        # Gather all four directions
        coords = target_coords[:, None, :] + self.neighbor_offsets[None, :, :]  # (B, 4, 2)
        valid = ((coords >= 0) & (coords < grid_size)).all(dim=-1)              # (B, 4) bool
        safe_coords = coords.clamp(0, grid_size - 1)
        batch_idx = torch.arange(B, device=device)[:, None]
        placed_cand = grid_placed[batch_idx, safe_coords[..., 0], safe_coords[..., 1]]  # (B, 4)
        is_placed = (placed_cand >= 0) & valid                                  # (B, 4) bool

        # Gather candidate features for placed neighbors
        safe_cand = placed_cand.clamp(min=0)
        cand_feats = raw_feats[batch_idx, safe_cand]                            # (B, 4, D)
        kv = cand_feats + self.dir_embed[None, :, :]                            # (B, 4, D)

        # Mask: True means masked (ignore)
        key_padding_mask = ~is_placed                                           # (B, 4) bool

        # Unmask a zeroed key for empty neighborhoods to prevent NaNs without a GPU sync.
        all_masked = key_padding_mask.all(dim=-1)                               # (B,)
        safe_mask = key_padding_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] & ~all_masked

        q = query_pos  # (B, 1, D)
        for layer in self.layers:
            # Pre-LN Cross Attention
            q_norm = layer["norm_q"](q)
            kv_norm = layer["norm_kv"](kv)
            attn_out, _ = layer["cross_attn"](
                query=q_norm,
                key=kv_norm,
                value=kv_norm,
                key_padding_mask=safe_mask,
                need_weights=False
            )
            q = q + attn_out

            # Pre-LN MLP
            q = q + layer["mlp"](layer["norm_mlp"](q))

        out = self.final_norm(q).squeeze(1)  # (B, D)

        # Zero out context for samples that had zero placed neighbors.
        out = out.masked_fill(all_masked[:, None], 0.0)

        return out


class GlobalBranch(nn.Module):
    """
    Global branch:
    4-layer, 8-head Pre-LN Transformer over all K candidates + target slot cross-attention.
    Supports PyTorch gradient checkpointing.
    """
    def __init__(
        self,
        content_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.content_dim = content_dim
        self.gradient_checkpointing = gradient_checkpointing

        # Embedding for unplaced candidates
        self.unplaced_embed = nn.Parameter(torch.randn(1, 1, content_dim) * 0.02)

        # 4-layer Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=content_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Cross attention for target empty slot to query global candidates
        self.norm_q = nn.LayerNorm(content_dim)
        self.norm_kv = nn.LayerNorm(content_dim)
        self.target_cross_attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(content_dim)
        self.mlp = nn.Sequential(
            nn.Linear(content_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, content_dim),
        )
        self.final_norm = nn.LayerNorm(content_dim)

    def run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer encoder with optional gradient checkpointing.
        x: (B, K, D)
        """
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            for layer in self.transformer.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            if self.transformer.norm is not None:
                x = self.transformer.norm(x)
            return x
        else:
            return self.transformer(x)

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
            H_cand: (B, K, D) globally contextualized candidate representations
        """
        B, K, D = raw_feats.shape

        # Build position/state embeddings for all K candidates
        safe_r = cand_to_slot[:, :, 0].clamp(0, pos_embed_table.shape[0] - 1)
        safe_c = cand_to_slot[:, :, 1].clamp(0, pos_embed_table.shape[1] - 1)
        placed_pos_embeds = pos_embed_table[safe_r, safe_c]  # (B, K, D)

        unplaced_embeds = self.unplaced_embed.expand(B, K, D)
        placed_mask = used_candidates.unsqueeze(-1)          # (B, K, 1)
        state_embeds = torch.where(placed_mask, placed_pos_embeds, unplaced_embeds)  # (B, K, D)

        transformer_input = raw_feats + state_embeds  # (B, K, D)
        H_cand = self.run_transformer(transformer_input)  # (B, K, D)

        # Target slot queries H_cand
        q_norm = self.norm_q(query_pos)
        kv_norm = self.norm_kv(H_cand)
        attn_out, _ = self.target_cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=False,
        )  # (B, 1, D)

        h = query_pos + attn_out
        h = h + self.mlp(self.norm_mlp(h))
        F_global = self.final_norm(h).squeeze(1)  # (B, D)

        return F_global, H_cand


class VirtualPatchModule(nn.Module):
    """
    Coordinate-free Virtual Patch Module:
    A learnable query vector Q_virtual (without any coordinate or directional embeddings)
    attends over the contextualized candidates H_cand of the fully-placed board
    via temperature-scaled softmax cosine similarity to produce a whole-image embedding Z_virtual.
    """
    def __init__(self, content_dim: int = 256, temperature: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.content_dim = content_dim
        self.temperature = temperature
        # Learnable virtual query token: coordinate-free
        self.q_virtual = nn.Parameter(torch.randn(1, 1, content_dim) * 0.02)
        self.query_norm = nn.LayerNorm(content_dim)
        self.context_norm = nn.LayerNorm(content_dim)
        self.cross_attn = nn.MultiheadAttention(content_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(content_dim)
        self.mlp = nn.Sequential(nn.Linear(content_dim, content_dim * 4), nn.GELU(),
                                 nn.Linear(content_dim * 4, content_dim))
        self.norm = nn.LayerNorm(content_dim)

    def forward(self, H_cand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H_cand: (B, K, D) candidate representations from final placed board
        Returns:
            z_virtual: (B, D) whole-image representation
            attn_weights: (B, K) attention weights over candidates
        """
        B, K, D = H_cand.shape
        q = self.q_virtual.expand(B, 1, D)  # (B, 1, D)
        context = self.context_norm(H_cand)
        attended, _ = self.cross_attn(self.query_norm(q), context, context, need_weights=False)
        q = q + attended
        q = q + self.mlp(self.mlp_norm(q))

        q_norm = F.normalize(q, p=2, dim=-1)           # (B, 1, D)
        h_norm = F.normalize(H_cand, p=2, dim=-1)      # (B, K, D)
        sim = torch.bmm(q_norm, h_norm.transpose(1, 2)).squeeze(1) / self.temperature  # (B, K)
        weights = F.softmax(sim, dim=-1)                # (B, K)

        z_virtual = torch.bmm(weights.unsqueeze(1), H_cand).squeeze(1)  # (B, D)
        z_virtual = self.norm(z_virtual)
        return z_virtual, weights


class JigsawSolverV3(nn.Module):
    """
    Complete v3 Model for Tiny ImageNet:
    - 3-stage Residual PatchCNNEncoder
    - 2-layer LocalBranch
    - 4-layer Pre-LN GlobalBranch with checkpointing
    - FuseMLP & Cosine Similarity Matching
    - Coordinate-free VirtualPatchModule for whole-image representation learning
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.content_dim = config.content_dim
        self.temperature = config.temperature

        # 2D Grid Position Embeddings table (16x16 to easily fit 5x5)
        max_grid = 16
        self.pos_embed = nn.Parameter(torch.randn(max_grid, max_grid, self.content_dim) * 0.02)

        # CNN Content Encoder
        self.encoder = PatchCNNEncoder(
            content_dim=self.content_dim,
            stages=config.cnn_stages
        )

        # Semantic branches
        self.local_branch = LocalBranch(
            content_dim=self.content_dim,
            num_heads=config.local_heads,
            num_layers=config.local_layers,
            mlp_ratio=2
        )
        self.global_branch = GlobalBranch(
            content_dim=self.content_dim,
            num_heads=config.global_heads,
            num_layers=config.global_layers,
            ffn_dim=config.global_ffn_dim,
            gradient_checkpointing=config.gradient_checkpointing
        )

        # Fuse MLP: [F_local (D), F_global (D), target_pos (D)] -> (3*D) -> D
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.content_dim * 3, self.content_dim),
            nn.GELU(),
            nn.Linear(self.content_dim, self.content_dim),
            nn.LayerNorm(self.content_dim)
        )

        # Coordinate-free Virtual Patch module
        self.virtual_patch = VirtualPatchModule(
            content_dim=self.content_dim,
            temperature=config.temperature,
            num_heads=config.global_heads
        )

    def encode_candidates(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode candidate patches once:
        patches: (B, K, 3, 13, 13)
        returns: (B, K, D)
        """
        return self.encoder(patches)

    def score_step(
        self,
        raw_feats: torch.Tensor,              # (B, K, D)
        grid_placed: torch.Tensor,            # (B, G, G) candidate index or -1
        used_candidates: torch.Tensor,        # (B, K) bool
        cand_to_slot: torch.Tensor,           # (B, K, 2) coords or -1
        target_coords: torch.Tensor,          # (B, 2) (r, c)
        grid_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores all K candidates for target empty slot.
        Returns:
            scores: (B, K) logits scaled by temperature
            H_cand: (B, K, D) current contextualized candidate representations
        """
        B, K, D = raw_feats.shape

        # Target slot position embedding: (B, 1, D)
        target_r = target_coords[:, 0].clamp(0, self.pos_embed.shape[0] - 1)
        target_c = target_coords[:, 1].clamp(0, self.pos_embed.shape[1] - 1)
        query_pos = self.pos_embed[target_r, target_c].unsqueeze(1)  # (B, 1, D)
        target_pos_flat = query_pos.squeeze(1)                        # (B, D)

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

        query_norm = F.normalize(Q_match, p=2, dim=-1)
        cand_norm = F.normalize(H_cand, p=2, dim=-1)
        scores = torch.bmm(cand_norm, query_norm.unsqueeze(-1)).squeeze(-1) / self.temperature  # (B, K)

        return scores, H_cand

    def compute_virtual_patch(
        self,
        raw_feats: torch.Tensor,              # (B, K, D)
        cand_to_slot: torch.Tensor,           # (B, K, 2) final placed coordinates
        used_candidates: torch.Tensor         # (B, K) bool, should be all True when fully placed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the coordinate-free virtual patch Z_virtual on the fully placed board.
        Returns:
            z_virtual: (B, D)
            attn_weights: (B, K)
        """
        # Run global branch transformer on the placed board
        safe_r = cand_to_slot[:, :, 0].clamp(0, self.pos_embed.shape[0] - 1)
        safe_c = cand_to_slot[:, :, 1].clamp(0, self.pos_embed.shape[1] - 1)
        placed_pos_embeds = self.pos_embed[safe_r, safe_c]  # (B, K, D)

        unplaced_embeds = self.global_branch.unplaced_embed.expand_as(placed_pos_embeds)
        placed_mask = used_candidates.unsqueeze(-1)
        state_embeds = torch.where(placed_mask, placed_pos_embeds, unplaced_embeds)

        H_final = self.global_branch.run_transformer(raw_feats + state_embeds)
        z_virtual, attn_weights = self.virtual_patch(H_final)
        return z_virtual, attn_weights
