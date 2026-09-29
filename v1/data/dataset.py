"""
CIFAR-10 数据集加载与训练/验证/测试划分
"""

import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T


class CIFAR10Dataset(Dataset):
    """
    加载 CIFAR-10 原始图像 (保持 [0, 1] 浮点张量)。
    """
    def __init__(self, root: str, train: bool = True):
        super().__init__()
        self.raw_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=False,
            transform=T.ToTensor()
        )

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, label = self.raw_dataset[idx] # (3, 32, 32)
        return {
            "img": img,
            "label": label,
            "img_id": idx
        }


def get_cifar_dataloaders(
    data_dir: str = "/home/cjc/桌面/myidea/data/cifar10",
    batch_size: int = 64,
    num_workers: int = 4,
    train_val_split: int = 45000,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    划分 CIFAR-10 为 45,000 训练集、5,000 验证集和 10,000 测试集。
    """
    full_train = CIFAR10Dataset(root=data_dir, train=True)
    test_ds = CIFAR10Dataset(root=data_dir, train=False)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices = indices[:train_val_split]
    val_indices = indices[train_val_split:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
