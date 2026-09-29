"""
单图拼图还原系统 (v1 版本) - 可视化脚本 (visualize.py)
对比展示: 原图 vs 打乱后状态 vs 模型还原拼图结果
并在三幅视图中均明确标注出：算法选取作为还原起点的初始中心块位置 (起点)。
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 配置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from v1.config import ModelConfig
from v1.data.dataset import get_cifar_dataloaders
from v1.models.puzzle_model import PuzzleModel


def parse_args():
    parser = argparse.ArgumentParser(description="单图拼图还原效果可视化 (显著标出起点位置)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/grid3/best_checkpoint.pth", help="模型权重文件")
    parser.add_argument("--data_dir", type=str, default="/home/cjc/桌面/myidea/data/cifar10", help="CIFAR-10 数据目录")
    parser.add_argument("--output_path", type=str, default="./visualizations/puzzle_results.png", help="输出图片路径")
    parser.add_argument("--num_samples", type=int, default=5, help="展示样本数量")
    parser.add_argument("--seed_mode", type=str, default="center", choices=["center", "random"], help="起点模式: center(中心位置块) 或 random(随机块)")
    parser.add_argument("--seed_r", type=int, default=None, help="自定义起点行坐标 (默认按 seed_mode)")
    parser.add_argument("--seed_c", type=int, default=None, help="自定义起点列坐标 (默认按 seed_mode)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    return parser.parse_args()


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) -> (H, W, 3) in [0, 1]"""
    return torch.clamp(t, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()


def draw_puzzle_annotations(ax, g: int, P: int, seed_r: int, seed_c: int, badge_text: str = "★起点"):
    """
    在图像上绘制拼图网格边界虚线，并以红金双层边框及文本角标显著标出【起点】。
    """
    # 1. 绘制各 Patch 之间的白色半透明网格分割虚线
    for r in range(g):
        for c in range(g):
            ax.add_patch(patches.Rectangle(
                (c * P - 0.5, r * P - 0.5), P, P,
                linewidth=0.6,
                edgecolor='white',
                facecolor='none',
                linestyle=':',
                alpha=0.35
            ))

    # 2. 绘制醒目的起点标记框 (外层鲜红 + 内层耀金)
    if seed_r is not None and seed_c is not None:
        outer_box = patches.Rectangle(
            (seed_c * P - 0.5, seed_r * P - 0.5), P, P,
            linewidth=2.8,
            edgecolor='#D32F2F', # 鲜红高对比外框
            facecolor='none'
        )
        inner_box = patches.Rectangle(
            (seed_c * P - 0.5, seed_r * P - 0.5), P, P,
            linewidth=1.6,
            edgecolor='#FFD700', # 耀金内框
            facecolor='none'
        )
        ax.add_patch(outer_box)
        ax.add_patch(inner_box)

        # 3. 添加显式文本角标 "★起点"，让观察者一眼识别起点位置
        tag_fontsize = 8 if g <= 3 else (6.5 if g == 5 else 5.5)
        ax.text(
            seed_c * P + 0.2, seed_r * P + 0.2,
            badge_text,
            color='white',
            fontsize=tag_fontsize,
            fontweight='bold',
            va='top',
            ha='left',
            bbox=dict(
                boxstyle='square,pad=0.12',
                facecolor='#D32F2F',
                edgecolor='#FFD700',
                linewidth=0.8,
                alpha=0.9
            )
        )


@torch.no_grad()
def run_visualization(args):
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    print(f"=== 载入检查点: {args.checkpoint} ===")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    grid_size = ckpt.get("grid_size", 3)
    g = grid_size

    model = PuzzleModel(ModelConfig(grid_size=grid_size)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    P = model.patch_size
    print(f"模型网格尺寸: {g}x{g} | 单块尺寸: {P}x{P} 像素 | 起点选择模式: {args.seed_mode}")

    _, _, test_loader = get_cifar_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.num_samples,
        num_workers=2
    )
    batch = next(iter(test_loader))
    imgs = batch["img"].to(device)

    # 确定起点坐标
    if args.seed_r is not None and args.seed_c is not None:
        seed_coord = (args.seed_r, args.seed_c)
    elif args.seed_mode == "center":
        seed_coord = (g // 2, g // 2)
    else:
        seed_coord = None  # 随机起点

    out = model(imgs, seed_coord=seed_coord, use_teacher_forcing=False)

    # 切片与未裁剪填充图像（保证三个视图的坐标系与 PxP 网格精确完全对齐）
    orig_patches = model.slicer.slice_image(imgs)
    orig_padded = model.slicer.unslice_image(orig_patches, crop_to_32=False).cpu()
    assembled_padded = model.slicer.unslice_image(out["grid_patches"], crop_to_32=False).cpu()

    shuffled_patches = out["shuffled_patches"].cpu()  # (B, K, 3, P, P)
    shuffled_grid = shuffled_patches.view(args.num_samples, g, g, 3, P, P).permute(0, 3, 1, 4, 2, 5).reshape(
        args.num_samples, 3, g * P, g * P
    )
    correct_mask = (out["all_preds"] == out["all_targets"]).cpu()
    perms = out["perms"].cpu()

    # 实际使用的起点物理坐标 (r0, c0)
    r0, c0 = out["seed_coord"]
    k0 = r0 * g + c0

    fig, axes = plt.subplots(args.num_samples, 3, figsize=(11, 3.4 * args.num_samples))
    if args.num_samples == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(
        f"{g}×{g} 单图拼图还原效果对比 (红金双层高亮框与角标明确标出：还原初始【起点】)",
        fontsize=13,
        fontweight="bold",
        y=0.995
    )

    for i in range(args.num_samples):
        # 计算打乱堆中起点块的具体槽位 (r_shuff, c_shuff)
        cand_idx = (perms[i] == k0).nonzero().item()
        r_shuff = cand_idx // g
        c_shuff = cand_idx % g

        # 1. 原始图像
        axes[i, 0].imshow(tensor_to_img(orig_padded[i]))
        axes[i, 0].set_title(f"样本 {i+1}: 原始图像 (★起点: [{r0}, {c0}])", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")
        draw_puzzle_annotations(axes[i, 0], g, P, r0, c0, badge_text="★起点")

        # 2. 打乱碎片候选网格
        axes[i, 1].imshow(tensor_to_img(shuffled_grid[i]))
        axes[i, 1].set_title(f"打乱碎片候选堆 (★起点打乱至: [{r_shuff}, {c_shuff}])", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")
        draw_puzzle_annotations(axes[i, 1], g, P, r_shuff, c_shuff, badge_text="★起点")

        # 3. 模型还原拼图结果
        is_perfect = correct_mask[i].all().item()
        acc = correct_mask[i].float().mean().item() * 100.0
        status_str = "完美还原 100%" if is_perfect else f"准确率: {acc:.1f}%"
        axes[i, 2].imshow(tensor_to_img(assembled_padded[i]))
        axes[i, 2].set_title(
            f"模型还原拼图 ({status_str}) (★起步锚点: [{r0}, {c0}])",
            fontsize=11,
            fontweight="bold",
            color="darkgreen" if is_perfect else "crimson"
        )
        axes[i, 2].axis("off")
        draw_puzzle_annotations(axes[i, 2], g, P, r0, c0, badge_text="★起点")

    plt.tight_layout(rect=[0, 0.01, 1, 0.985])
    plt.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"★ 拼图对比效果图已成功生成至: {args.output_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_visualization(cli_args)
