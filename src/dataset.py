"""
Animals90 Dataset Module
数据加载和预处理模块
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class Animals90Dataset(Dataset):
    """Animals90数据集类"""

    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (str): 数据集根目录
            transform (callable, optional): 数据转换操作
            is_train (bool): 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # 获取所有类别文件夹
        self.classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 加载所有图片路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图片
        image = Image.open(img_path).convert('RGB')

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_train=True, img_size=224):
    """
    获取数据转换操作

    Args:
        is_train (bool): 是否为训练集
        img_size (int): 图片大小

    Returns:
        transforms.Compose: 转换操作组合
    """
    if is_train:
        # 训练集数据增强
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证集/测试集转换
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def create_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4, img_size=224):
    """
    创建训练和验证数据加载器

    Args:
        train_dir (str): 训练集目录
        val_dir (str): 验证集目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        img_size (int): 图片大小

    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # 创建数据集
    train_dataset = Animals90Dataset(
        root_dir=train_dir,
        transform=get_transforms(is_train=True, img_size=img_size),
        is_train=True
    )

    val_dataset = Animals90Dataset(
        root_dir=val_dir,
        transform=get_transforms(is_train=False, img_size=img_size),
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, num_classes, train_dataset.classes
