"""
DeepSpeed Training Launcher
简化的DeepSpeed启动脚本
"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Launch DeepSpeed training')

    # 基本参数
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='验证集目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')

    # DeepSpeed参数
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='GPU数量')
    parser.add_argument('--fp16', action='store_true',
                        help='使用FP16混合精度')
    parser.add_argument('--offload_optimizer', action='store_true',
                        help='将优化器卸载到CPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数')

    args = parser.parse_args()

    # 构建DeepSpeed命令
    cmd = [
        'deepspeed',
        f'--num_gpus={args.num_gpus}',
        'src/train_deepspeed.py',
        f'--train_dir={args.train_dir}',
        f'--val_dir={args.val_dir}',
        f'--epochs={args.epochs}',
        f'--batch_size={args.batch_size}',
        f'--learning_rate={args.learning_rate}',
        f'--num_workers={args.num_workers}',
        f'--gradient_accumulation_steps={args.gradient_accumulation_steps}'
    ]

    if args.fp16:
        cmd.append('--fp16')

    if args.offload_optimizer:
        cmd.append('--offload_optimizer')

    print("=" * 60)
    print("DeepSpeed ZeRO-2 Training Launcher")
    print("=" * 60)
    print(f"GPUs: {args.num_gpus}")
    print(f"Batch Size (per GPU): {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"FP16: {args.fp16}")
    print(f"Offload Optimizer: {args.offload_optimizer}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print("=" * 60)
    print(f"\nCommand: {' '.join(cmd)}\n")

    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
