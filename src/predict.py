"""
Inference Script for Animals90 Classification
推理/预测脚本
"""

import os
import argparse
import json
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

from model import ResNet18Animals90


class Animals90Predictor:
    """Animals90预测器"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path (str): 检查点文件路径
            device (str): 设备类型
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载检查点
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 获取类别信息
        self.num_classes = checkpoint['num_classes']
        self.classes = checkpoint['classes']
        print(f"Number of classes: {self.num_classes}")

        # 创建模型
        self.model = ResNet18Animals90(num_classes=self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 数据转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print("Model loaded successfully!")

    def predict_image(self, image_path, top_k=5):
        """
        预测单张图片

        Args:
            image_path (str): 图片路径
            top_k (int): 返回top-k预测结果

        Returns:
            list: [(class_name, probability), ...]
        """
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # 获取top-k结果
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.classes[idx]
            results.append((class_name, float(prob)))

        return results

    def predict_batch(self, image_paths):
        """
        批量预测图片

        Args:
            image_paths (list): 图片路径列表

        Returns:
            list: 每张图片的预测结果
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append({
                'image_path': image_path,
                'predictions': result
            })
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict animals using trained ResNet18 model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点文件路径')
    parser.add_argument('--image', type=str,
                        help='单张图片路径')
    parser.add_argument('--image_dir', type=str,
                        help='图片目录（批量预测）')
    parser.add_argument('--top_k', type=int, default=5,
                        help='显示top-k预测结果')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备类型')
    parser.add_argument('--output', type=str,
                        help='结果保存路径（JSON格式）')

    args = parser.parse_args()

    # 创建预测器
    predictor = Animals90Predictor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 单张图片预测
    if args.image:
        print(f"\nPredicting: {args.image}")
        print("-" * 60)

        results = predictor.predict_image(args.image, top_k=args.top_k)

        print(f"\nTop-{args.top_k} Predictions:")
        for i, (class_name, prob) in enumerate(results, 1):
            print(f"  {i}. {class_name}: {prob * 100:.2f}%")

        if args.output:
            output_data = {
                'image_path': args.image,
                'predictions': [{'class': c, 'probability': p} for c, p in results]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"\nResults saved to: {args.output}")

    # 批量预测
    elif args.image_dir:
        print(f"\nBatch predicting from: {args.image_dir}")
        print("-" * 60)

        # 获取所有图片
        image_paths = []
        for filename in os.listdir(args.image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(args.image_dir, filename))

        print(f"Found {len(image_paths)} images")

        # 批量预测
        all_results = []
        for image_path in image_paths:
            results = predictor.predict_image(image_path, top_k=args.top_k)
            all_results.append({
                'image_path': image_path,
                'predictions': [{'class': c, 'probability': p} for c, p in results]
            })

            # 显示结果
            print(f"\n{os.path.basename(image_path)}:")
            for i, (class_name, prob) in enumerate(results[:3], 1):
                print(f"  {i}. {class_name}: {prob * 100:.2f}%")

        # 保存结果
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"\n\nAll results saved to: {args.output}")

    else:
        print("Error: Please provide either --image or --image_dir")


if __name__ == '__main__':
    main()
