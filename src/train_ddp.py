"""
DDP Training Script for Animals90 Classification
使用PyTorch DDP（Distributed Data Parallel）分布式训练
支持混合精度和梯度累积
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# 修复PyTorch MUSA兼容性问题 - 直接在storage类上添加属性
import torch

# 为所有storage类添加is_musa属性
for storage_class in [
    torch.storage.UntypedStorage,
    torch.FloatStorage, torch.DoubleStorage, torch.HalfStorage,
    torch.BFloat16Storage, torch.LongStorage, torch.IntStorage,
    torch.ShortStorage, torch.CharStorage, torch.ByteStorage, torch.BoolStorage
]:
    if not hasattr(storage_class, 'is_musa'):
        storage_class.is_musa = property(lambda self: False)

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model import ResNet18Animals90
from dataset import Animals90Dataset, get_transforms



class DDPTrainer:
    """DDP训练器类"""

    def __init__(self, args, config):
        """
        Args:
            args: 命令行参数
            config (dict): 配置参数
        """
        self.config = config
        self.args = args
        self.local_rank = args.local_rank
        self.is_main_process = self.local_rank == 0

        # 初始化分布式环境
        if args.distributed:
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.world_size = dist.get_world_size()
        else:
            # 单GPU或CPU训练
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
            self.world_size = 1

        if self.is_main_process:
            print(f"Using device: {self.device}")
            print(f"World size: {self.world_size}")
            print(f"Mixed Precision (AMP): {args.use_amp}")

        # 创建数据集
        if self.is_main_process:
            print("Loading datasets...")

        train_dataset = Animals90Dataset(
            root_dir=config['train_dir'],
            transform=get_transforms(is_train=True, img_size=config['img_size']),
            is_train=True
        )

        val_dataset = Animals90Dataset(
            root_dir=config['val_dir'],
            transform=get_transforms(is_train=False, img_size=config['img_size']),
            is_train=False
        )

        self.num_classes = len(train_dataset.classes)
        self.classes = train_dataset.classes

        if self.is_main_process:
            print(f"Number of classes: {self.num_classes}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")

        # 创建数据加载器
        if args.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # 创建模型
        if self.is_main_process:
            print("\nCreating model...")

        model = ResNet18Animals90(
            num_classes=self.num_classes,
            pretrained=config['pretrained'],
            freeze_backbone=config['freeze_backbone']
        )
        model = model.to(self.device)

        # 使用DDP包装模型
        if args.distributed:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        else:
            self.model = model

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=self.is_main_process
        )

        # 混合精度训练
        self.scaler = GradScaler() if args.use_amp else None

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
        if self.is_main_process:
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            os.makedirs(config['log_dir'], exist_ok=True)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 设置epoch以确保每个epoch的shuffle不同
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]")
        else:
            pbar = self.train_loader

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # 混合精度前向传播
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            if self.is_main_process and isinstance(pbar, tqdm):
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

        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Val]")
        else:
            pbar = self.val_loader

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                # 混合精度推理
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                if self.is_main_process and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': running_loss / (batch_idx + 1),
                        'acc': 100. * correct / total
                    })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        if not self.is_main_process:
            return

        # 获取模型状态（处理DDP包装）
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
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
        if self.is_main_process:
            print("\n" + "=" * 60)
            print("Starting DDP Training")
            print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate(epoch)

            # 只在主进程更新学习率和保存
            if self.is_main_process:
                # 记录历史
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rate'].append(current_lr)

                # 更新学习率
                self.scheduler.step(val_loss)

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
        if self.is_main_process:
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

        # 清理分布式环境
        if self.args.distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Animals90 with DDP')

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
                        help='每GPU批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # DDP参数
    parser.add_argument('--local_rank', type=int, default=0,
                        help='DDP本地rank')
    parser.add_argument('--distributed', action='store_true',
                        help='使用分布式训练')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')

    # 保存参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志保存目录')

    args = parser.parse_args()

    # 配置字典
    config = {
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'img_size': args.img_size,
        'pretrained': args.pretrained,
        'freeze_backbone': args.freeze_backbone,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir
    }

    # 创建训练器并开始训练
    trainer = DDPTrainer(args, config)
    trainer.train()


if __name__ == '__main__':
    main()
