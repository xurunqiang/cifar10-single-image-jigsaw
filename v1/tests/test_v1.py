"""
单图拼图还原系统 (v1 版本) 单元测试
覆盖切片还原、模型模块、前向传播、反向传播与过拟合能力
"""

import unittest
import torch
from v1.config import ModelConfig
from v1.data.patch_slicer import PatchSlicer
from v1.models.content_encoder import ContentEncoder
from v1.models.local_fusion import LocalFusion
from v1.models.global_fusion import GlobalFusion
from v1.models.puzzle_model import PuzzleModel


class TestV1Pipeline(unittest.TestCase):
    def test_slicer_and_unslice(self):
        for g, (p, exp_h) in {3: (11, 33), 5: (7, 35), 7: (5, 35)}.items():
            slicer = PatchSlicer(grid_size=g)
            x = torch.randn(2, 3, 32, 32)
            patches = slicer.slice_image(x)
            self.assertEqual(patches.shape, (2, g, g, 3, p, p))

            # 重组还原回 32x32 物理图像
            unslice = slicer.unslice_image(patches, crop_to_32=True)
            self.assertEqual(unslice.shape, (2, 3, 32, 32))
            # 原始原图重组应该与原图严格一致
            self.assertTrue(torch.allclose(x, unslice, atol=1e-5))

    def test_content_encoder(self):
        enc = ContentEncoder(content_dim=96)
        patch = torch.randn(4, 3, 11, 11)
        feat = enc(patch)
        self.assertEqual(feat.shape, (4, 96))

    def test_fusion_modules(self):
        loc = LocalFusion(dim=96)
        glob = GlobalFusion(dim=96, grid_size=3)
        grid_feat = torch.randn(2, 96, 3, 3)
        grid_mask = torch.ones(2, 1, 3, 3)

        fused_dict = loc(grid_feat, grid_mask, 1, 1)
        self.assertIn((1, 1), fused_dict)
        self.assertEqual(fused_dict[(1, 1)].shape, (2, 96))

        z_glob, slot_tokens = glob(grid_feat)
        self.assertEqual(z_glob.shape, (2, 96))
        self.assertEqual(slot_tokens.shape, (2, 9, 96))

    def test_puzzle_model_forward_and_backward_3x3(self):
        cfg = ModelConfig(grid_size=3, content_dim=96)
        model = PuzzleModel(cfg=cfg)
        imgs = torch.randn(4, 3, 32, 32)

        out = model(imgs)
        self.assertIn("loss", out)
        self.assertIn("loss_step", out)
        self.assertIn("loss_global", out)
        self.assertIn("patch_acc", out)
        self.assertIn("puzzle_acc", out)
        self.assertIn("assembled_img", out)
        self.assertEqual(out["assembled_img"].shape, (4, 3, 32, 32))

        # 测试反向传播: 确保 LocalFusion 和 GlobalFusion 均能收到梯度
        loss = out["loss"]
        loss.backward()
        
        # 验证全局融合模块有梯度
        gf_has_grad = any(p.grad is not None and (p.grad.abs() > 0).any() for p in model.global_fusion.parameters())
        self.assertTrue(gf_has_grad, "GlobalFusion 必须收到反向传播梯度！")

        # 验证局部融合模块有梯度
        lf_has_grad = any(p.grad is not None and (p.grad.abs() > 0).any() for p in model.local_fusion.parameters())
        self.assertTrue(lf_has_grad, "LocalFusion 必须收到反向传播梯度！")

    def test_puzzle_model_forward_5x5(self):
        cfg = ModelConfig(grid_size=5, content_dim=96)
        model = PuzzleModel(cfg=cfg)
        imgs = torch.randn(2, 3, 32, 32)
        out = model(imgs)
        self.assertEqual(out["assembled_img"].shape, (2, 3, 32, 32))
        self.assertEqual(out["all_logits"].shape[1], 24) # 25 - 1 = 24 步


if __name__ == "__main__":
    unittest.main()
