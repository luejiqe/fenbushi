"""
DDP Training Launcher
PyTorch DDP分布式训练启动器
"""

import subprocess
import sys
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Launch DDP training')

    # 基本参数
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='验证集目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每GPU批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # DDP参数
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='GPU数量')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')

    args = parser.parse_args()

    # 检测GPU
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU training")
        num_gpus = 0
    else:
        available_gpus = torch.cuda.device_count()
        num_gpus = min(args.num_gpus, available_gpus)
        if args.num_gpus > available_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available")

    print("=" * 60)
    print("PyTorch DDP Training Launcher")
    print("=" * 60)
    print(f"GPUs: {num_gpus}")
    print(f"Batch Size (per GPU): {args.batch_size}")
    print(f"Effective Batch Size: {args.batch_size * max(1, num_gpus)}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Mixed Precision (AMP): {args.use_amp}")
    print(f"Num Workers: {args.num_workers}")
    print("=" * 60)

    if num_gpus > 1:
        # 多GPU分布式训练
        cmd = [
            'torchrun',
            f'--nproc_per_node={num_gpus}',
            '--nnodes=1',
            '--node_rank=0',
            '--master_addr=localhost',
            '--master_port=29500',
            'src/train_ddp.py',
            f'--train_dir={args.train_dir}',
            f'--val_dir={args.val_dir}',
            f'--epochs={args.epochs}',
            f'--batch_size={args.batch_size}',
            f'--learning_rate={args.learning_rate}',
            f'--num_workers={args.num_workers}',
            '--distributed'
        ]
    else:
        # 单GPU训练
        cmd = [
            'python',
            'src/train_ddp.py',
            f'--train_dir={args.train_dir}',
            f'--val_dir={args.val_dir}',
            f'--epochs={args.epochs}',
            f'--batch_size={args.batch_size}',
            f'--learning_rate={args.learning_rate}',
            f'--num_workers={args.num_workers}'
        ]

    if args.use_amp:
        cmd.append('--use_amp')

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
