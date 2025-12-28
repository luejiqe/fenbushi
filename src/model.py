"""
ResNet18 Model for Animals90 Classification
基于ResNet18的动物分类模型
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Animals90(nn.Module):
    """基于ResNet18的Animals90分类模型"""

    def __init__(self, num_classes=90, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            pretrained (bool): 是否使用预训练权重
            freeze_backbone (bool): 是否冻结主干网络
        """
        super(ResNet18Animals90, self).__init__()

        # 加载预训练的ResNet18模型
        if pretrained:
            self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet18 = models.resnet18(weights=None)

        # 冻结主干网络参数（可选）
        if freeze_backbone:
            for param in self.resnet18.parameters():
                param.requires_grad = False

        # 获取ResNet18最后一层的输入特征数
        num_features = self.resnet18.fc.in_features

        # 替换全连接层以适应Animals90数据集
        self.resnet18.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入图片张量 [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: 分类logits [batch_size, num_classes]
        """
        return self.resnet18(x)

    def unfreeze_backbone(self):
        """解冻主干网络，允许微调"""
        for param in self.resnet18.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """冻结主干网络参数"""
        for name, param in self.resnet18.named_parameters():
            if 'fc' not in name:  # 不冻结最后的分类层
                param.requires_grad = False


def create_model(num_classes=90, pretrained=True, freeze_backbone=False, device='cuda'):
    """
    创建模型实例

    Args:
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用预训练权重
        freeze_backbone (bool): 是否冻结主干网络
        device (str): 设备类型 ('cuda' 或 'cpu')

    Returns:
        ResNet18Animals90: 模型实例
    """
    model = ResNet18Animals90(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )

    # 移动到指定设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model


if __name__ == '__main__':
    # 测试模型
    model = create_model(num_classes=90, pretrained=True, device='cpu')
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
