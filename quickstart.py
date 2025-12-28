"""
Quick Start Script
快速开始脚本 - 用于测试模型和数据流
"""

import torch
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import create_model, ResNet18Animals90
from dataset import get_transforms


def test_model():
    """测试模型创建和前向传播"""
    print("=" * 60)
    print("Testing Model Creation and Forward Pass")
    print("=" * 60)

    # 创建模型
    print("\n1. Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_classes=90, pretrained=True, device=device)
    print(f"✓ Model created successfully on {device}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ Forward pass successful!")

    # 测试数据转换
    print("\n4. Testing data transforms...")
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    print(f"   ✓ Train transform: {len(train_transform.transforms)} steps")
    print(f"   ✓ Val transform: {len(val_transform.transforms)} steps")

    print("\n" + "=" * 60)
    print("All Tests Passed! ✓")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Prepare your data in data/train and data/val directories")
    print("2. Run training: python src/train.py --train_dir data/train --val_dir data/val")
    print("3. Check README.md for more details")


def check_environment():
    """检查环境配置"""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)

    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # 检查CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        # 检查MUSA（摩尔线程GPU）
        try:
            import torch_musa
            print(f"MUSA available: {torch_musa.is_available()}")
            if torch_musa.is_available():
                device_name = torch_musa.get_device_name(0) if hasattr(torch_musa, 'get_device_name') else 'Moore Threads GPU'
                print(f"MUSA device: {device_name}")
                print(f"MUSA count: {torch_musa.device_count() if hasattr(torch_musa, 'device_count') else 1}")
        except ImportError:
            print(f"MUSA available: False (torch_musa not installed)")

    # 检查目录
    print("\nDirectory structure:")
    dirs = ['data/train', 'data/val', 'data/test', 'checkpoints', 'logs']
    for d in dirs:
        exists = "✓" if os.path.exists(d) else "✗"
        print(f"   {exists} {d}")


if __name__ == '__main__':
    check_environment()
    test_model()
