"""
Evaluation Script for Animals90 Classification
模型评估脚本
"""

import os
import argparse
import torch

from model import ResNet18Animals90
from dataset import create_dataloaders
from utils import evaluate_model, plot_confusion_matrix, save_classification_report, count_parameters


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet18 on Animals90 dataset')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点文件路径')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='测试集目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备类型')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='结果保存目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载检查点
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    num_classes = checkpoint['num_classes']
    classes = checkpoint['classes']
    print(f"Number of classes: {num_classes}")

    # 创建模型
    print("\nCreating model...")
    model = ResNet18Animals90(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 统计参数
    param_stats = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {param_stats['total_parameters']:,}")
    print(f"  Trainable: {param_stats['trainable_parameters']:,}")
    print(f"  Non-trainable: {param_stats['non_trainable_parameters']:,}")

    # 创建数据加载器
    print(f"\nLoading test dataset from: {args.test_dir}")
    from dataset import Animals90Dataset, get_transforms

    test_dataset = Animals90Dataset(
        root_dir=args.test_dir,
        transform=get_transforms(is_train=False),
        is_train=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")

    # 评估模型
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    results = evaluate_model(model, test_loader, device, classes)

    # 打印结果
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")

    # 保存分类报告
    report_path = os.path.join(args.output_dir, 'classification_report.json')
    save_classification_report(results['classification_report'], report_path)

    # 打印Top-5类别准确率
    print("\nTop-5 Best Performing Classes:")
    class_accuracies = []
    for class_name in classes:
        if class_name in results['classification_report']:
            acc = results['classification_report'][class_name]['precision'] * 100
            class_accuracies.append((class_name, acc))

    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    for i, (class_name, acc) in enumerate(class_accuracies[:5], 1):
        print(f"  {i}. {class_name}: {acc:.2f}%")

    print("\nTop-5 Worst Performing Classes:")
    for i, (class_name, acc) in enumerate(class_accuracies[-5:], 1):
        print(f"  {i}. {class_name}: {acc:.2f}%")

    # 绘制混淆矩阵
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], classes, save_path=cm_path)

    print("\n" + "=" * 60)
    print("Evaluation Completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
