"""
DeepSpeed Training Launcher for Dual RTX 5090
优化用于双卡NVIDIA RTX 5090 (24GB * 2)
"""

import subprocess
import sys
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Launch DeepSpeed training for dual RTX 5090')

    # 基本参数
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='验证集目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='每GPU批次大小 (推荐: 64-128 for RTX 5090)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载线程数 (推荐: 8-16 for RTX 5090)')

    # DeepSpeed参数
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='GPU数量 (默认: 2 for dual RTX 5090)')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='使用FP16混合精度 (强烈推荐)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--config', type=str, default='ds_config_rtx5090.json',
                        help='DeepSpeed配置文件')

    # 高级参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_deepspeed',
                        help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='保存检查点间隔')

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        print(f"请使用 ds_config_rtx5090.json 或 ds_config_zero2.json")
        sys.exit(1)

    # 构建DeepSpeed命令
    cmd = [
        'deepspeed',
        f'--num_gpus={args.num_gpus}',
        f'--master_port=29500',
        'src/train_deepspeed.py',
        f'--deepspeed_config={args.config}',
        f'--train_dir={args.train_dir}',
        f'--val_dir={args.val_dir}',
        f'--epochs={args.epochs}',
        f'--batch_size={args.batch_size}',
        f'--learning_rate={args.learning_rate}',
        f'--num_workers={args.num_workers}',
        f'--gradient_accumulation_steps={args.gradient_accumulation_steps}',
        f'--checkpoint_dir={args.checkpoint_dir}',
        f'--save_interval={args.save_interval}'
    ]

    if args.fp16:
        cmd.append('--fp16')

    # 计算有效批次大小
    effective_batch_size = args.batch_size * args.num_gpus * args.gradient_accumulation_steps

    print("=" * 80)
    print("DeepSpeed ZeRO-2 Training Launcher - Dual RTX 5090 Optimized")
    print("=" * 80)
    print(f"🎮 GPUs: {args.num_gpus} x NVIDIA RTX 5090 (24GB)")
    print(f"📦 Batch Size per GPU: {args.batch_size}")
    print(f"📊 Effective Batch Size: {effective_batch_size}")
    print(f"🔄 Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"📚 Epochs: {args.epochs}")
    print(f"📖 Learning Rate: {args.learning_rate}")
    print(f"⚡ FP16 Mixed Precision: {'✅ Enabled' if args.fp16 else '❌ Disabled'}")
    print(f"🔧 Num Workers: {args.num_workers}")
    print(f"⚙️  DeepSpeed Config: {args.config}")
    print(f"💾 Checkpoint Dir: {args.checkpoint_dir}")
    print("=" * 80)

    # 显存估算
    estimated_vram = args.batch_size * 0.15  # 粗略估算每个样本约150MB
    print(f"\n📊 估算显存使用: ~{estimated_vram:.1f}GB per GPU (含ZeRO-2优化)")

    if estimated_vram > 22:
        print("⚠️  警告: 批次大小可能过大，建议减小 batch_size 或增加 gradient_accumulation_steps")
    elif estimated_vram < 10:
        print("💡 提示: 显存利用率较低，可以增加 batch_size 以提升训练速度")
    else:
        print("✅ 批次大小设置合理")

    print("\n" + "=" * 80)
    print(f"执行命令:\n{' '.join(cmd)}")
    print("=" * 80 + "\n")

    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 错误: 训练失败，退出代码 {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(0)
    except FileNotFoundError:
        print("\n❌ 错误: 找不到 deepspeed 命令")
        print("请确保已安装 DeepSpeed: pip install deepspeed")
        sys.exit(1)


if __name__ == '__main__':
    main()
