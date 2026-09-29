"""
单图拼图还原系统 (v1 版本) - 测试集评估程序 (evaluate.py)
在测试集上评测 Patch 还原命中率、整图完美还原率以及每步拓展的准确率变化。
"""

import os
import sys
import argparse
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from v1.config import ModelConfig
from v1.data.dataset import get_cifar_dataloaders
from v1.models.puzzle_model import PuzzleModel


def parse_args():
    parser = argparse.ArgumentParser(description="评估单图拼图还原效果")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/grid3/best_checkpoint.pth", help="模型权重文件")
    parser.add_argument("--data_dir", type=str, default="/home/cjc/桌面/myidea/data/cifar10", help="CIFAR-10 数据目录")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    return parser.parse_args()


@torch.no_grad()
def evaluate_test(args):
    device = torch.device(args.device)
    print(f"=== 载入检查点: {args.checkpoint} ===")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    grid_size = ckpt.get("grid_size", 3)
    model_cfg = ModelConfig(grid_size=grid_size)
    model = PuzzleModel(cfg=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"模型网格尺寸: {grid_size}x{grid_size} | 设备: {device}")

    _, _, test_loader = get_cifar_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    total_loss = 0.0
    total_patch_correct = 0
    total_patch_count = 0
    total_puzzle_correct = 0
    total_images = 0

    num_steps = grid_size * grid_size - 1
    step_correct = torch.zeros(num_steps)
    step_count = torch.zeros(num_steps)

    for b_idx, batch in enumerate(test_loader):
        imgs = batch["img"].to(device)
        B = imgs.shape[0]

        out = model(imgs, use_teacher_forcing=False)
        total_loss += out["loss"].item() * B

        correct_mask = (out["all_preds"] == out["all_targets"]).cpu() # (B, S)
        total_patch_correct += correct_mask.sum().item()
        total_patch_count += correct_mask.numel()

        total_puzzle_correct += correct_mask.all(dim=1).sum().item()
        total_images += B

        step_correct += correct_mask.sum(dim=0).float()
        step_count += B

    avg_loss = total_loss / total_images
    patch_acc = (total_patch_correct / total_patch_count) * 100.0
    puzzle_acc = (total_puzzle_correct / total_images) * 100.0

    print("\n================ 测试集综合评估报告 ================")
    print(f"测试集样本总数: {total_images}")
    print(f"平均交叉熵损失: {avg_loss:.4f}")
    print(f"★ Patch 平均还原命中率: {patch_acc:.2f}%")
    print(f"★ 整图完全拼对率 (完美复原): {puzzle_acc:.2f}%")
    print("---------------- 每步组装命中率分布 ----------------")
    step_accs = (step_correct / step_count) * 100.0
    for s_idx, acc in enumerate(step_accs):
        print(f"  第 {s_idx+1:2d} 块拼装命中率: {acc:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate_test(cli_args)
