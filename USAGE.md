# Animals90 动物识别系统 - 使用示例

本文档提供详细的使用示例和最佳实践。

## 快速开始

### 1. 环境测试

首先运行快速测试脚本，确保环境配置正确：

```bash
python quickstart.py
```

这将检查：
- Python和PyTorch版本
- CUDA是否可用
- 目录结构是否正确
- 模型是否能正常创建和运行

## 完整使用流程

### Step 1: 准备数据

假设你已经下载了Animals90数据集，需要将其组织成以下结构：

```bash
# 创建数据目录
mkdir -p data/{train,val,test}

# 将数据移动到对应目录
# 每个类别一个文件夹，例如：
# data/train/cat/img1.jpg
# data/train/dog/img2.jpg
# ...
```

### Step 2: 训练模型

#### 基础训练（推荐新手）

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 30 \
    --batch_size 32
```

#### 高级训练（推荐有经验用户）

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --scheduler cosine \
    --weight_decay 0.0001 \
    --img_size 224 \
    --num_workers 8
```

#### 迁移学习两阶段训练

**阶段1: 冻结主干网络，只训练分类头**

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --freeze_backbone \
    --checkpoint_dir checkpoints/stage1
```

**阶段2: 解冻主干网络，微调整个模型**

```bash
# 需要修改代码加载stage1的检查点
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --checkpoint_dir checkpoints/stage2
```

### Step 3: 监控训练

训练过程中会显示：
- 每个epoch的训练和验证损失
- 训练和验证准确率
- 当前学习率
- 最佳验证准确率

训练完成后，可以可视化训练历史：

```bash
python src/utils.py \
    --history logs/training_history.json \
    --output training_curves.png
```

### Step 4: 评估模型

在测试集上评估模型性能：

```bash
python src/evaluate.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --test_dir data/test \
    --batch_size 64 \
    --output_dir evaluation_results
```

这将生成：
- `evaluation_results/classification_report.json` - 详细分类报告
- `evaluation_results/confusion_matrix.png` - 混淆矩阵可视化

### Step 5: 推理预测

#### 单张图片预测

```bash
python src/predict.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --image path/to/cat.jpg \
    --top_k 5
```

输出示例：
```
Top-5 Predictions:
  1. cat: 98.45%
  2. kitten: 1.23%
  3. tiger: 0.18%
  4. leopard: 0.08%
  5. lion: 0.06%
```

#### 批量预测

```bash
python src/predict.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --image_dir path/to/images/ \
    --top_k 3 \
    --output predictions.json
```

## 训练技巧和最佳实践

### 1. 学习率选择

- **初始训练**: 0.001 (Adam) 或 0.01 (SGD)
- **微调**: 0.0001 - 0.0005
- **冻结主干**: 可以使用更大的学习率如 0.01

### 2. 批次大小建议

根据GPU显存选择：
- 8GB GPU: batch_size = 32
- 12GB GPU: batch_size = 64
- 16GB GPU: batch_size = 128
- 24GB GPU: batch_size = 256

### 3. 数据增强策略

训练集增强（已实现）：
- RandomCrop
- RandomHorizontalFlip
- RandomRotation
- ColorJitter

如需更强的增强，可以修改 [src/dataset.py](src/dataset.py) 添加：
- RandomVerticalFlip
- RandomAffine
- GaussianBlur
- RandomErasing

### 4. 过拟合处理

如果出现过拟合（训练acc高但验证acc低）：
- 增加Dropout比率
- 增大weight_decay
- 使用更强的数据增强
- 收集更多训练数据
- 早停

### 5. 欠拟合处理

如果出现欠拟合（训练和验证acc都低）：
- 增加训练轮数
- 使用更大的学习率
- 减小weight_decay
- 使用更复杂的模型（ResNet50/101）
- 减弱数据增强

## 常见场景示例

### 场景1: 快速原型验证

```bash
# 小批次，少轮数，快速验证
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 5 \
    --batch_size 16 \
    --img_size 128
```

### 场景2: 追求最高精度

```bash
# 大批次，多轮数，精细调参
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.0003 \
    --scheduler cosine \
    --weight_decay 0.0001 \
    --img_size 256
```

### 场景3: CPU训练

```bash
# 减小批次和图片大小
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 20 \
    --batch_size 8 \
    --img_size 128 \
    --num_workers 2
```

## 性能优化

### 1. 数据加载优化

- 增加 `num_workers` (通常设为CPU核心数)
- 使用 `pin_memory=True` (已实现)
- 预处理数据并缓存

### 2. 训练加速

- 使用混合精度训练 (需添加AMP支持)
- 梯度累积（小显存时）
- 分布式训练（多GPU）

### 3. 推理优化

- 模型量化
- ONNX导出
- TensorRT加速
- 批量推理

## 故障排除

### 问题1: CUDA out of memory

**解决方案**:
```bash
# 减小batch_size
--batch_size 16

# 或减小图片大小
--img_size 128

# 或两者都减小
--batch_size 8 --img_size 112
```

### 问题2: 训练速度慢

**解决方案**:
- 增加 `num_workers`
- 使用SSD存储数据
- 减小图片分辨率
- 使用更小的模型

### 问题3: 准确率不提升

**检查清单**:
- [ ] 数据标注是否正确
- [ ] 类别分布是否平衡
- [ ] 学习率是否合适
- [ ] 是否过拟合
- [ ] 数据增强是否合理

## 进阶功能

### 自定义数据增强

编辑 [src/dataset.py](src/dataset.py):

```python
# 添加更多增强
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.1),  # 新增
    transforms.GaussianBlur(kernel_size=3),  # 新增
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),  # 新增
])
```

### 使用其他模型

编辑 [src/model.py](src/model.py) 替换ResNet18为其他模型：

```python
from torchvision import models

# ResNet50
self.backbone = models.resnet50(pretrained=True)

# EfficientNet
self.backbone = models.efficientnet_b0(pretrained=True)

# Vision Transformer
self.backbone = models.vit_b_16(pretrained=True)
```

## 联系与支持

如有问题，请提交Issue或查看README.md。
