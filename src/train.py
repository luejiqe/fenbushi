"""
Training Script for Animals90 Classification
训练脚本
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from model import create_model
from dataset import create_dataloaders


class Trainer:
    """训练器类"""

    def __init__(self, config):
        """
        Args:
            config (dict): 配置参数
        """
        self.config = config

        # 自动检测设备：优先CUDA，其次MUSA（摩尔线程），最后CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            print(f"Using device: cuda")
            print(f"GPU: {device_name}")
        else:
            # 尝试检测MUSA（摩尔线程GPU）
            try:
                import torch_musa
                if torch_musa.is_available():
                    self.device = torch.device('musa')
                    device_name = torch_musa.get_device_name(0) if hasattr(torch_musa, 'get_device_name') else 'Moore Threads GPU'
                    print(f"Using device: musa")
                    print(f"GPU: {device_name}")
                else:
                    self.device = torch.device('cpu')
                    print(f"Using device: cpu")
            except ImportError:
                self.device = torch.device('cpu')
                print(f"Using device: cpu")

        # 创建数据加载器
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.num_classes, self.classes = create_dataloaders(
            train_dir=config['train_dir'],
            val_dir=config['val_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            img_size=config['img_size']
        )

        print(f"Number of classes: {self.num_classes}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        # 创建模型
        print("\nCreating model...")
        self.model = create_model(
            num_classes=self.num_classes,
            pretrained=config['pretrained'],
            freeze_backbone=config['freeze_backbone'],
            device=self.device
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

        # 创建保存目录
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Val]")
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'config': self.config
        }

        # 保存最新的检查点
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            'checkpoint_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with accuracy: {val_acc:.2f}%")

    def train(self):
        """训练主循环"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate(epoch)

            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 打印结果
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch

            self.save_checkpoint(epoch, val_acc, is_best)

            print(f"  Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
            print("-" * 60)

        # 训练结束
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Completed!")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print("=" * 60)

        # 保存训练历史
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"\nTraining history saved to: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Animals90 dataset')

    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='验证集目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图片大小')

    # 模型参数
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='冻结主干网络')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='学习率调度器')

    # 保存参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志保存目录')

    args = parser.parse_args()

    # 配置字典
    config = vars(args)

    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
