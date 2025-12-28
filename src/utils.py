"""
Utility Functions for Animals90 Project
工具函数
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np
from tqdm import tqdm


def plot_training_history(history_path, save_path=None):
    """
    绘制训练历史曲线

    Args:
        history_path (str): 训练历史JSON文件路径
        save_path (str): 保存图片路径
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy曲线
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Learning Rate曲线
    axes[2].plot(history['learning_rate'], label='Learning Rate', marker='o', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def evaluate_model(model, dataloader, device, classes):
    """
    评估模型性能

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        classes: 类别列表

    Returns:
        dict: 评估结果
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算准确率
    accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)

    # 分类报告
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, classes, save_path=None, figsize=(20, 20)):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        classes: 类别列表
        save_path: 保存路径
        figsize: 图片大小
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def save_classification_report(report, save_path):
    """
    保存分类报告

    Args:
        report: 分类报告字典
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {save_path}")


def count_parameters(model):
    """
    统计模型参数量

    Args:
        model: 模型

    Returns:
        dict: 参数统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


if __name__ == '__main__':
    # 测试绘制训练历史
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, help='训练历史JSON路径')
    parser.add_argument('--output', type=str, help='输出图片路径')
    args = parser.parse_args()

    if args.history:
        plot_training_history(args.history, args.output)
