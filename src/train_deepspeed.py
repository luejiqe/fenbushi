"""
DeepSpeed Training Script for Animals90 Classification
使用DeepSpeed ZeRO-2加速训练
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm import tqdm

from model import ResNet18Animals90
from dataset import create_dataloaders


class DeepSpeedTrainer:
    """DeepSpeed训练器类"""

    def __init__(self, args, config):
        """
        Args:
            args: DeepSpeed参数
            config (dict): 配置参数
        """
        self.config = config
        self.args = args

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
        model = ResNet18Animals90(
            num_classes=self.num_classes,
            pretrained=config['pretrained'],
            freeze_backbone=config['freeze_backbone']
        )

        # DeepSpeed配置
        ds_config = {
            "train_batch_size": config['batch_size'] * args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": config['batch_size'],
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": config['learning_rate'],
                    "weight_decay": config['weight_decay']
                }
            },
            "fp16": {
                "enabled": args.fp16
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu" if args.offload_optimizer else "none",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            }
        }

        # 初始化DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        self.device = self.model_engine.local_rank
        print(f"Using DeepSpeed with device: {self.device}")
        print(f"ZeRO Stage: 2")
        print(f"FP16: {args.fp16}")
        print(f"Offload Optimizer: {args.offload_optimizer}")

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

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
        self.model_engine.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.model_engine.local_rank)
            labels = labels.to(self.model_engine.local_rank)

            # 前向传播
            outputs = self.model_engine(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.model_engine.backward(loss)
            self.model_engine.step()

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
        self.model_engine.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Val]")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.model_engine.local_rank)
                labels = labels.to(self.model_engine.local_rank)

                # 前向传播
                outputs = self.model_engine(images)
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
        # DeepSpeed保存检查点
        checkpoint_dir = os.path.join(
            self.config['checkpoint_dir'],
            f'epoch_{epoch}'
        )
        self.model_engine.save_checkpoint(checkpoint_dir, tag=f'epoch_{epoch}')

        # 保存额外信息
        extra_info = {
            'epoch': epoch,
            'val_acc': val_acc,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'config': self.config
        }
        extra_path = os.path.join(checkpoint_dir, 'extra_info.json')
        with open(extra_path, 'w') as f:
            json.dump(extra_info, f, indent=4)

        # 保存最佳模型
        if is_best:
            best_dir = os.path.join(self.config['checkpoint_dir'], 'best_model')
            self.model_engine.save_checkpoint(best_dir, tag='best')
            best_extra_path = os.path.join(best_dir, 'extra_info.json')
            with open(best_extra_path, 'w') as f:
                json.dump(extra_info, f, indent=4)
            print(f"✓ Saved best model with accuracy: {val_acc:.2f}%")

    def train(self):
        """训练主循环"""
        print("\n" + "=" * 60)
        print("Starting DeepSpeed Training (ZeRO-2)")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate(epoch)

            # 记录历史
            current_lr = self.model_engine.get_lr()[0]
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

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

            if epoch % self.config.get('save_interval', 10) == 0 or is_best:
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
    parser = argparse.ArgumentParser(description='Train ResNet18 on Animals90 with DeepSpeed')

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

    # DeepSpeed参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DeepSpeed本地rank')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='使用FP16混合精度训练')
    parser.add_argument('--offload_optimizer', action='store_true', default=False,
                        help='将优化器状态卸载到CPU')

    # 保存参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_deepspeed',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点的间隔轮数')

    # 解析参数
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
        'log_dir': args.log_dir,
        'save_interval': args.save_interval
    }

    # 初始化DeepSpeed
    deepspeed.init_distributed()

    # 创建训练器并开始训练
    trainer = DeepSpeedTrainer(args, config)
    trainer.train()


if __name__ == '__main__':
    main()
