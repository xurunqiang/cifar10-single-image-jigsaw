"""
单图拼图还原系统 (v1 版本) - 训练程序 (train.py)
支持 3x3, 5x5, 7x7 网格，具备 tqdm 实时进度条、每轮验证集正确率输出、最佳模型保存与断点续训能力。
"""

import os
import sys
import time
import argparse
import tempfile
from dataclasses import asdict
from typing import Dict, Any

# 将根目录添加到 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from v1.config import ModelConfig, TrainConfig
from v1.data.dataset import get_cifar_dataloaders
from v1.models.puzzle_model import PuzzleModel


def parse_args():
    parser = argparse.ArgumentParser(description="训练单图拼图还原网络 (v1 版本)")
    parser.add_argument("--data_dir", type=str, default="/home/cjc/桌面/myidea/data/cifar10", help="CIFAR-10 数据目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="检查点保存根目录")
    parser.add_argument("--grid_size", type=int, default=3, choices=[3, 5, 7], help="拼图网格尺寸 (3, 5, 7)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--epochs", type=int, default=100, help="总训练轮数 (默认 100 轮)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="从指定检查点文件恢复训练 (断点续训)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--quick_test", action="store_true", help="冒烟测试模式")
    return parser.parse_args()


def save_checkpoint_atomic(state: Dict[str, Any], path: str):
    """原子化保存检查点，写完临时文件再覆盖，防止中断损坏。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path) or '.', suffix='.tmp', delete=False) as handle:
        temporary = handle.name
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@torch.no_grad()
def evaluate_validation(model: PuzzleModel, val_loader, device: torch.device, max_batches: int = None) -> Dict[str, float]:
    """
    在验证集上评估损失、Patch 级别命中率和整图完美拼对率 (带进度条)。
    """
    model.eval()
    total_loss = 0.0
    total_patch_correct = 0
    total_patch_count = 0
    total_puzzle_correct = 0
    total_images = 0

    val_pbar = tqdm(val_loader, desc="[Validating]", dynamic_ncols=True, leave=False)
    for b_idx, batch in enumerate(val_pbar):
        if max_batches is not None and b_idx >= max_batches:
            break
        imgs = batch["img"].to(device)
        B = imgs.shape[0]

        out = model(imgs, use_teacher_forcing=False)
        total_loss += out["loss"].item() * B

        correct_mask = (out["all_preds"] == out["all_targets"]) # (B, S)
        total_patch_correct += correct_mask.sum().item()
        total_patch_count += correct_mask.numel()

        total_puzzle_correct += correct_mask.all(dim=1).sum().item()
        total_images += B

        val_pbar.set_postfix({
            "val_loss": f"{out['loss'].item():.3f}",
            "patch_acc": f"{(total_patch_correct / total_patch_count) * 100.0:.1f}%"
        })

    avg_loss = total_loss / max(1, total_images)
    patch_acc = (total_patch_correct / max(1, total_patch_count)) * 100.0
    puzzle_acc = (total_puzzle_correct / max(1, total_images)) * 100.0

    return {
        "val_loss": avg_loss,
        "val_patch_acc": patch_acc,
        "val_puzzle_acc": puzzle_acc
    }


def train_v1(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    save_dir = os.path.join(args.save_dir, f"grid{args.grid_size}")
    os.makedirs(save_dir, exist_ok=True)

    print("================================================================================")
    print(f"★ 启动单图拼图还原系统 (v1 版本) 训练")
    print(f"  计算设备: {device} | 网格尺寸: {args.grid_size}x{args.grid_size} (共 {args.grid_size**2} 块)")
    print(f"  批次大小: {args.batch_size} | 学习率: {args.lr} | 目标轮数: {args.epochs}")
    print(f"  检查点保存路径: {save_dir}")
    print("================================================================================")

    # 1. 数据准备
    epochs = 2 if args.quick_test else args.epochs
    train_loader, val_loader, test_loader = get_cifar_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2 if args.quick_test else args.num_workers,
        seed=args.seed
    )

    # 2. 模型构建
    model_cfg = ModelConfig(grid_size=args.grid_size)
    model = PuzzleModel(cfg=model_cfg).to(device)

    # 3. 优化器与学习率调度
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(5, max(1, epochs // 10))
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6),
    ], milestones=[warmup_epochs])

    best_val_patch_acc = -1.0
    start_epoch = 1

    # 4. 断点续训处理
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n>>> 正在恢复断点检查点: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_patch_acc = ckpt.get("best_val_patch_acc", ckpt.get("val_patch_acc", -1.0))
            if "rng_state" in ckpt:
                torch.set_rng_state(ckpt["rng_state"].cpu())
            if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
            print(f"★ 成功恢复！将从第 {start_epoch} 轮继续训练 (当前最佳验证集命中率: {best_val_patch_acc:.2f}%)\n")
        else:
            raise FileNotFoundError(f"未找到指定的恢复检查点: {args.resume}")

    # 5. 训练与验证循环
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_patch_correct = 0
        running_patch_count = 0
        running_puzzle_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:3d}/{epochs:3d}]", dynamic_ncols=True)

        for b_idx, batch in enumerate(pbar):
            imgs = batch["img"].to(device)
            B = imgs.shape[0]

            out = model(imgs, use_teacher_forcing=False)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 统计指标
            with torch.no_grad():
                running_loss += loss.item() * B
                correct_mask = (out["all_preds"] == out["all_targets"])
                running_patch_correct += correct_mask.sum().item()
                running_patch_count += correct_mask.numel()
                running_puzzle_correct += correct_mask.all(dim=1).sum().item()
                total_samples += B

                cur_patch_acc = (running_patch_correct / running_patch_count) * 100.0

            # 动态刷新进度条信息
            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Step": f"{out['loss_step'].item():.3f}",
                "Glob": f"{out['loss_global'].item():.3f}",
                "Acc": f"{cur_patch_acc:.1f}%",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })

            if args.quick_test and b_idx >= 3:
                break

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # 训练集整体平均指标
        train_avg_loss = running_loss / max(1, total_samples)
        train_patch_acc = (running_patch_correct / max(1, running_patch_count)) * 100.0
        train_puzzle_acc = (running_puzzle_correct / max(1, total_samples)) * 100.0

        # 验证集评估指标 (带验证集进度条并计算命中率)
        val_res = evaluate_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            max_batches=5 if args.quick_test else None
        )

        # 格式化终端输出
        print(f"\n==================== Epoch [{epoch:3d}/{epochs:3d}] 轮次汇总 (耗时 {epoch_time:.2f}s) ====================")
        print(f"  [训练集] 损失: {train_avg_loss:.4f} | Patch 命中率: {train_patch_acc:6.2f}% | 整图完美率: {train_puzzle_acc:5.2f}%")
        print(f"  [验证集] 损失: {val_res['val_loss']:.4f} | Patch 命中率: {val_res['val_patch_acc']:6.2f}% | 整图完美率: {val_res['val_puzzle_acc']:5.2f}%")

        # 检查是否刷新最佳记录
        is_best = val_res["val_patch_acc"] > best_val_patch_acc
        if is_best:
            best_val_patch_acc = val_res["val_patch_acc"]

        # 打包检查点状态
        ckpt = {
            "epoch": epoch,
            "grid_size": args.grid_size,
            "model_config": asdict(model_cfg),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_patch_acc": val_res["val_patch_acc"],
            "val_puzzle_acc": val_res["val_puzzle_acc"],
            "best_val_patch_acc": best_val_patch_acc,
            "val_loss": val_res["val_loss"],
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "args": vars(args)
        }

        # 随时保存最新检查点
        save_checkpoint_atomic(ckpt, os.path.join(save_dir, "latest_checkpoint.pth"))

        if is_best:
            save_checkpoint_atomic(ckpt, os.path.join(save_dir, "best_checkpoint.pth"))
            print(f"  ★ 最佳检查点已更新并保存至 best_checkpoint.pth (验证集命中率: {best_val_patch_acc:.2f}%)")
        print("========================================================================================\n")

    print("\n🎉 训练全部顺利完成！")


if __name__ == "__main__":
    args = parse_args()
    train_v1(args)
