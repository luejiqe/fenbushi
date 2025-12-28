"""
Dataset Split Tool
数据集划分工具 - 将训练集按比例划分为训练集、验证集和测试集
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    将数据集划分为训练集、验证集和测试集

    Args:
        source_dir (str): 原始数据目录（包含所有类别文件夹）
        target_dir (str): 目标数据目录（将创建train/val/test子目录）
        train_ratio (float): 训练集比例 (默认: 0.7)
        val_ratio (float): 验证集比例 (默认: 0.15)
        test_ratio (float): 测试集比例 (默认: 0.15)
        seed (int): 随机种子，保证可复现
    """
    # 检查比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1.0，当前: {train_ratio + val_ratio + test_ratio}"

    # 设置随机种子
    random.seed(seed)

    # 创建目标目录
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有类别
    classes = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

    print(f"\n发现 {len(classes)} 个类别")
    print(f"划分比例 - 训练集: {train_ratio*100}%, 验证集: {val_ratio*100}%, 测试集: {test_ratio*100}%")
    print("=" * 80)

    # 统计信息
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }

    # 处理每个类别
    for class_name in tqdm(classes, desc="处理类别"):
        class_path = os.path.join(source_dir, class_name)

        # 获取该类别的所有图片
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # 打乱图片顺序
        random.shuffle(images)

        # 计算划分点
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        # 划分图片
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # 创建类别目录
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # 复制文件到训练集
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
            stats['train'][class_name] += 1

        # 复制文件到验证集
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)
            stats['val'][class_name] += 1

        # 复制文件到测试集
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
            stats['test'][class_name] += 1

    # 打印统计信息
    print("\n" + "=" * 80)
    print("数据集划分完成！")
    print("=" * 80)

    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    total_all = total_train + total_val + total_test

    print(f"\n总体统计:")
    print(f"  训练集: {total_train:,} 张 ({total_train/total_all*100:.1f}%)")
    print(f"  验证集: {total_val:,} 张 ({total_val/total_all*100:.1f}%)")
    print(f"  测试集: {total_test:,} 张 ({total_test/total_all*100:.1f}%)")
    print(f"  总计:   {total_all:,} 张")

    # 打印每个类别的统计（可选）
    print(f"\n各类别分布:")
    print(f"{'类别':<20} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}")
    print("-" * 80)

    for class_name in sorted(classes):
        train_count = stats['train'][class_name]
        val_count = stats['val'][class_name]
        test_count = stats['test'][class_name]
        total_count = train_count + val_count + test_count

        print(f"{class_name:<20} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")

    print("\n" + "=" * 80)
    print(f"数据已保存到:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  测试集: {test_dir}")
    print("=" * 80)


def move_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    移动文件而不是复制（节省空间）

    警告：此操作会移动原始文件，请谨慎使用！
    """
    # 检查比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1.0，当前: {train_ratio + val_ratio + test_ratio}"

    # 设置随机种子
    random.seed(seed)

    # 创建目标目录
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有类别
    classes = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

    print(f"\n发现 {len(classes)} 个类别")
    print(f"划分比例 - 训练集: {train_ratio*100}%, 验证集: {val_ratio*100}%, 测试集: {test_ratio*100}%")
    print("⚠️  警告: 使用移动模式，将移动原始文件！")
    print("=" * 80)

    # 统计信息
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }

    # 处理每个类别
    for class_name in tqdm(classes, desc="处理类别"):
        class_path = os.path.join(source_dir, class_name)

        # 获取该类别的所有图片
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # 打乱图片顺序
        random.shuffle(images)

        # 计算划分点
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        # 划分图片
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # 创建类别目录
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # 移动文件到训练集
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.move(src, dst)
            stats['train'][class_name] += 1

        # 移动文件到验证集
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.move(src, dst)
            stats['val'][class_name] += 1

        # 移动文件到测试集
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            shutil.move(src, dst)
            stats['test'][class_name] += 1

    # 打印统计信息
    print("\n" + "=" * 80)
    print("数据集划分完成！")
    print("=" * 80)

    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    total_all = total_train + total_val + total_test

    print(f"\n总体统计:")
    print(f"  训练集: {total_train:,} 张 ({total_train/total_all*100:.1f}%)")
    print(f"  验证集: {total_val:,} 张 ({total_val/total_all*100:.1f}%)")
    print(f"  测试集: {total_test:,} 张 ({total_test/total_all*100:.1f}%)")
    print(f"  总计:   {total_all:,} 张")

    print("\n" + "=" * 80)
    print(f"数据已保存到:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  测试集: {test_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='数据集划分工具 - 将数据集划分为训练集、验证集和测试集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 复制模式（推荐，保留原始数据）:
   python split_dataset.py --source data/raw --target data --train 0.7 --val 0.15 --test 0.15

2. 移动模式（节省空间，但会移动原始文件）:
   python split_dataset.py --source data/raw --target data --train 0.7 --val 0.15 --test 0.15 --move

3. 自定义随机种子:
   python split_dataset.py --source data/raw --target data --seed 123

4. 常见划分比例:
   - 70/15/15: --train 0.7 --val 0.15 --test 0.15
   - 80/10/10: --train 0.8 --val 0.1 --test 0.1
   - 60/20/20: --train 0.6 --val 0.2 --test 0.2
        """
    )

    parser.add_argument('--source', type=str, required=True,
                        help='原始数据目录（包含所有类别文件夹）')
    parser.add_argument('--target', type=str, default='data',
                        help='目标数据目录（默认: data）')
    parser.add_argument('--train', type=float, default=0.7,
                        help='训练集比例（默认: 0.7）')
    parser.add_argument('--val', type=float, default=0.15,
                        help='验证集比例（默认: 0.15）')
    parser.add_argument('--test', type=float, default=0.15,
                        help='测试集比例（默认: 0.15）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--move', action='store_true',
                        help='使用移动模式而不是复制模式（会移动原始文件）')

    args = parser.parse_args()

    # 检查源目录是否存在
    if not os.path.exists(args.source):
        print(f"错误: 源目录不存在: {args.source}")
        return

    # 确认操作
    print("\n" + "=" * 80)
    print("数据集划分配置:")
    print("=" * 80)
    print(f"源目录:     {args.source}")
    print(f"目标目录:   {args.target}")
    print(f"训练集:     {args.train * 100}%")
    print(f"验证集:     {args.val * 100}%")
    print(f"测试集:     {args.test * 100}%")
    print(f"随机种子:   {args.seed}")
    print(f"模式:       {'移动' if args.move else '复制'}")
    print("=" * 80)

    if args.move:
        print("\n⚠️  警告: 你选择了移动模式，这将移动原始文件！")
        response = input("确认继续？(输入 'yes' 继续): ")
        if response.lower() != 'yes':
            print("操作已取消")
            return

    # 执行划分
    if args.move:
        move_dataset(
            source_dir=args.source,
            target_dir=args.target,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )
    else:
        split_dataset(
            source_dir=args.source,
            target_dir=args.target,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
