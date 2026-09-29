"""
单图拼图还原模型 (PuzzleModel - v1 版本)
整合切片、单图打乱、真实物理位置种子锚定、BFS 队列四周扩展、
局部十字注意力融合、全局 Transformer 融合以及槽位分类预测。
"""

from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from v1.config import ModelConfig
from v1.data.patch_slicer import PatchSlicer
from v1.models.content_encoder import ContentEncoder
from v1.models.local_fusion import LocalFusion
from v1.models.global_fusion import GlobalFusion


class PuzzleModel(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.grid_size = self.cfg.grid_size
        self.content_dim = self.cfg.content_dim
        self.temperature = self.cfg.temperature

        self.slicer = PatchSlicer(grid_size=self.grid_size)
        self.patch_size = self.slicer.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 核心模块
        self.content_encoder = ContentEncoder(content_dim=self.content_dim)
        self.local_fusion = LocalFusion(
            dim=self.content_dim,
            num_heads=self.cfg.local_heads,
            mlp_ratio=self.cfg.local_mlp_ratio
        )
        self.global_fusion = GlobalFusion(
            dim=self.content_dim,
            grid_size=self.grid_size,
            num_layers=self.cfg.global_layers,
            num_heads=self.cfg.global_heads,
            mlp_ratio=self.cfg.global_mlp_ratio
        )

        # 目标槽位上下文与坐标投影头
        self.slot_pos_proj = nn.Sequential(
            nn.Linear(2, self.content_dim),
            nn.GELU(),
            nn.Linear(self.content_dim, self.content_dim)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(self.content_dim * 2, self.content_dim),
            nn.GELU(),
            nn.Linear(self.content_dim, self.content_dim)
        )

        # 正交 4 邻居偏移: 上、右、下、左
        self.nbr_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def forward(
        self,
        imgs: torch.Tensor,
        seed_coord: Optional[Tuple[int, int]] = None,
        use_teacher_forcing: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播: 单图打乱并逐步还原
        输入:
            imgs: (B, 3, 32, 32)
            seed_coord: 可选指定的真实种子坐标 (r, c)，为 None 则随机挑选一个
            use_teacher_forcing: 训练时若为 True 则每步填入真实正确 Patch，否则填入预测 Patch
        输出:
            dict 包含:
                loss: 平均交叉熵损失
                patch_acc: Patch 级别平均准确率
                puzzle_acc: 整图完美复原率 (所有槽位全部预测正确的比例)
                assembled_img: 拼装还原的物理图像 (B, 3, 32, 32)
                z_global: 全局融合语义向量 (B, 96)
        """
        B = imgs.shape[0]
        g = self.grid_size
        K = self.num_patches
        P = self.patch_size
        device = imgs.device

        # 1. 切片获取原始真实 Patch (B, g, g, 3, P, P)
        orig_patches_grid = self.slicer.slice_image(imgs)
        orig_patches = orig_patches_grid.reshape(B, K, 3, P, P)

        # 2. 编码所有原始 Patch
        flat_patches = orig_patches.reshape(B * K, 3, P, P)
        patch_feats = self.content_encoder(flat_patches).reshape(B, K, self.content_dim) # (B, K, 96)

        # 3. 单图内部打乱 Patch (每个样本独立生成排列 perm)
        perms = torch.stack([torch.randperm(K, device=device) for _ in range(B)], dim=0) # (B, K)
        shuffled_patches = torch.gather(
            orig_patches, 1, perms[:, :, None, None, None].expand(-1, -1, 3, P, P)
        ) # (B, K, 3, P, P)
        shuffled_feats = torch.gather(
            patch_feats, 1, perms[:, :, None].expand(-1, -1, self.content_dim)
        ) # (B, K, 96)
        orig_ids = perms # (B, K): shuffled[b, j] 对应的原图 patch id 是 perms[b, j]

        # 4. 随机选取中心种子块，并放置在真实的物理网格坐标上
        if seed_coord is None:
            r0 = torch.randint(0, g, (1,), device=device).item()
            c0 = torch.randint(0, g, (1,), device=device).item()
        else:
            r0, c0 = seed_coord
            assert 0 <= r0 < g and 0 <= c0 < g

        k0 = r0 * g + c0 # 真实属于 (r0, c0) 的 Patch ID
        # 寻找打乱列表里哪一个位置是 k0
        seed_cand_idx = (orig_ids == k0).nonzero(as_tuple=False)[:, 1] # (B,)

        # 5. 初始化网格画布
        grid_features = torch.zeros((B, self.content_dim, g, g), device=device)
        grid_mask = torch.zeros((B, 1, g, g), device=device)
        grid_patches = torch.zeros((B, g, g, 3, P, P), device=device)

        # 放入种子块到其真实的物理位置 (r0, c0)
        b_idx = torch.arange(B, device=device)
        grid_features[:, :, r0, c0] = shuffled_feats[b_idx, seed_cand_idx]
        grid_patches[:, r0, c0] = shuffled_patches[b_idx, seed_cand_idx]
        grid_mask[:, :, r0, c0] = 1.0

        # 可用候选块掩码 (B, K): True 为可用，已分配给其他已就位槽位的设为 False
        # 始终追踪哪些原图 patch 已经被就位 (无论预测是否正确，保证未处理槽位的真值永远在候选池内)
        processed_targets = torch.zeros((B, K), dtype=torch.bool, device=device)
        processed_targets[b_idx, k0] = True

        # 6. BFS 队列逐步向外扩展
        queue = deque([(r0, c0)])
        queued_set = {(r0, c0)}

        step_logits_list = []
        step_targets_list = []
        step_preds_list = []

        denom = max(1, g - 1)

        while queue:
            curr_r, curr_c = queue.popleft()

            # 遍历当前出队中心四周的 4 个正交邻居
            for dr, dc in self.nbr_offsets:
                nr, nc = curr_r + dr, curr_c + dc

                # 越界检查
                if not (0 <= nr < g and 0 <= nc < g):
                    continue
                # 若已填充则跳过
                if (nr, nc) in queued_set:
                    continue

                # 目标槽位真值 Patch ID 与打乱候选中的对应索引
                target_k = nr * g + nc
                target_cand_idx = (orig_ids == target_k).nonzero(as_tuple=False)[:, 1] # (B,)

                # 提取 (nr, nc) 周围已就位的有效正交邻居特征
                nbr_feat_list = []
                for cdr, cdc in self.nbr_offsets:
                    adj_r, adj_c = nr + cdr, nc + cdc
                    if 0 <= adj_r < g and 0 <= adj_c < g and (adj_r, adj_c) in queued_set:
                        nbr_feat_list.append(grid_features[:, :, adj_r, adj_c])

                nbr_context = torch.stack(nbr_feat_list, dim=1).mean(dim=1) # (B, 96)

                # 槽位归一化 2D 坐标嵌入
                norm_coord = torch.tensor([nr / denom, nc / denom], device=device, dtype=torch.float32)
                slot_pos = self.slot_pos_proj(norm_coord).unsqueeze(0).expand(B, -1) # (B, 96)

                # 融合生成目标槽位 Query
                query = self.query_mlp(torch.cat([nbr_context, slot_pos], dim=-1)) # (B, 96)
                query = F.normalize(query, dim=-1)

                # 计算与候选池中所有碎片的匹配分数
                norm_cands = F.normalize(shuffled_feats, dim=-1) # (B, K, 96)
                logits = torch.bmm(norm_cands, query.unsqueeze(-1)).squeeze(-1) / self.temperature # (B, K)

                # 仅屏蔽已就位槽位的候选，未就位槽位的真值候选绝对合法可用
                cand_is_already_placed = torch.gather(processed_targets, 1, orig_ids) # (B, K)
                available_mask = ~cand_is_already_placed
                masked_logits = logits.masked_fill(~available_mask, -1e4)
                pred_idx = masked_logits.argmax(dim=-1) # (B,)

                step_logits_list.append(masked_logits)
                step_targets_list.append(target_cand_idx)
                step_preds_list.append(pred_idx)

                # 选块填入网格 (训练时可开启 teacher forcing 或使用预测块)
                chosen_idx = target_cand_idx if (self.training and use_teacher_forcing) else pred_idx
                grid_features[:, :, nr, nc] = shuffled_feats[b_idx, chosen_idx]
                grid_patches[:, nr, nc] = shuffled_patches[b_idx, chosen_idx]
                grid_mask[:, :, nr, nc] = 1.0

                # 标记该槽位已处理
                processed_targets[b_idx, target_k] = True

                # 将新槽位加入已填集合与队列
                queued_set.add((nr, nc))
                queue.append((nr, nc))

            # 对当前出队中心及其四周新入驻邻居执行局部十字融合更新
            fused = self.local_fusion(grid_features, grid_mask, curr_r, curr_c, valid_coords=queued_set)
            grid_features = grid_features.clone()
            for (r, c), feat in fused.items():
                grid_features[:, :, r, c] = feat

        # 7. 全网格填满后，执行全局 Transformer 融合与全图位置核验预测 (实现全局注意力反向传播)
        z_global, slot_tokens = self.global_fusion(grid_features) # z_global: (B, 96), slot_tokens: (B, K, 96)

        # 全局上下文槽位核验打分 (由全局 Transformer 统揽整图后，核验每个网格位置属于哪个候选碎片)
        norm_slot_tokens = F.normalize(slot_tokens, dim=-1) # (B, K, 96)
        global_logits = torch.bmm(norm_slot_tokens, norm_cands.transpose(1, 2)) / self.temperature # (B, K, K)
        # 每个网格槽位真实对应的打乱候选索引即为 perms 的逆置换:
        target_cand_indices = torch.argsort(perms, dim=1) # (B, K)

        # 8. 拼装还原物理图像 (裁剪掉边缘填充，恢复 32x32)
        assembled_img = self.slicer.unslice_image(grid_patches, crop_to_32=True)

        # 9. 统计逐步损失 (LocalFusion 通路) + 全局损失 (GlobalFusion 通路)
        all_logits = torch.stack(step_logits_list, dim=1)   # (B, S, K), S = K - 1
        all_targets = torch.stack(step_targets_list, dim=1) # (B, S)
        all_preds = torch.stack(step_preds_list, dim=1)     # (B, S)

        loss_step = F.cross_entropy(all_logits.view(-1, K), all_targets.view(-1))
        loss_global = F.cross_entropy(global_logits.view(-1, K), target_cand_indices.view(-1))
        loss = loss_step + loss_global

        correct_mask = (all_preds == all_targets) # (B, S)
        patch_acc = correct_mask.float().mean()
        puzzle_acc = correct_mask.all(dim=1).float().mean()

        return {
            "loss": loss,
            "loss_step": loss_step,
            "loss_global": loss_global,
            "patch_acc": patch_acc,
            "puzzle_acc": puzzle_acc,
            "all_logits": all_logits,
            "all_targets": all_targets,
            "all_preds": all_preds,
            "global_logits": global_logits,
            "assembled_img": assembled_img,
            "z_global": z_global,
            "seed_coord": (r0, c0),
            "orig_patches": orig_patches,
            "shuffled_patches": shuffled_patches,
            "grid_patches": grid_patches,
            "perms": perms
        }
