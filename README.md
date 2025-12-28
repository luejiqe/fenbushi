# Animals90 动物识别系统

基于ResNet18的Animals90数据集动物识别系统，使用PyTorch实现。

## 项目结构

```
fenbushi/
├── src/                      # 源代码目录
│   ├── dataset.py           # 数据加载和预处理
│   ├── model.py             # ResNet18模型定义
│   ├── train.py             # 训练脚本
│   ├── predict.py           # 推理脚本
│   ├── evaluate.py          # 评估脚本
│   └── utils.py             # 工具函数
├── data/                     # 数据目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── checkpoints/             # 模型检查点
├── logs/                    # 训练日志
├── models/                  # 保存的模型
├── requirements.txt         # 依赖包
└── README.md               # 项目说明
```

## 功能特性

- ✅ 基于预训练ResNet18的迁移学习
- ✅ 支持90种动物分类
- ✅ 完整的数据增强策略
- ✅ 训练过程可视化
- ✅ 学习率自动调整
- ✅ 模型评估和性能分析
- ✅ 单张/批量图片推理
- ✅ 混淆矩阵和分类报告生成

## 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

## 安装

1. 克隆项目

```bash
git clone <repository-url>
cd fenbushi
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将Animals90数据集按以下结构组织：

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

每个类别对应一个文件夹，文件夹名即为类别名。

## 使用方法

### 1. 训练模型

基本训练命令：

```bash
python src/train.py --train_dir data/train --val_dir data/val
```

高级训练选项：

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --img_size 224 \
    --pretrained \
    --scheduler plateau \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

参数说明：
- `--train_dir`: 训练集目录
- `--val_dir`: 验证集目录
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批次大小 (默认: 32)
- `--learning_rate`: 初始学习率 (默认: 0.001)
- `--img_size`: 输入图片大小 (默认: 224)
- `--pretrained`: 使用ImageNet预训练权重
- `--freeze_backbone`: 冻结主干网络
- `--scheduler`: 学习率调度器 (plateau/cosine/none)
- `--weight_decay`: 权重衰减 (默认: 1e-4)
- `--num_workers`: 数据加载线程数 (默认: 4)

### 2. 模型推理

单张图片预测：

```bash
python src/predict.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --image path/to/image.jpg \
    --top_k 5
```

批量预测：

```bash
python src/predict.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --image_dir path/to/images/ \
    --top_k 5 \
    --output predictions.json
```

### 3. 模型评估

```bash
python src/evaluate.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --test_dir data/test \
    --batch_size 32 \
    --output_dir evaluation_results
```

评估结果包括：
- 整体准确率
- 每个类别的precision、recall、f1-score
- 混淆矩阵可视化
- 分类报告JSON文件

### 4. 可视化训练历史

```bash
python src/utils.py \
    --history logs/training_history.json \
    --output training_curves.png
```

## 模型架构

本项目使用ResNet18作为骨干网络，主要特点：

1. **预训练**: 使用ImageNet预训练权重进行迁移学习
2. **分类头**: 自定义全连接层
   - Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Linear(90)
3. **参数量**: 约11M参数

## 数据增强策略

训练集增强：
- 随机裁剪 (224x224)
- 随机水平翻转 (p=0.5)
- 随机旋转 (±15°)
- 颜色抖动 (亮度、对比度、饱和度、色调)
- 标准化 (ImageNet均值和标准差)

验证/测试集：
- 直接resize到224x224
- 标准化

## 训练技巧

1. **学习率调度**:
   - ReduceLROnPlateau: 当验证损失停止下降时降低学习率
   - CosineAnnealingLR: 余弦退火学习率

2. **正则化**:
   - Dropout层防止过拟合
   - 权重衰减 (L2正则化)

3. **早停策略**:
   - 自动保存最佳验证准确率的模型

## 性能优化建议

1. **增加训练数据**: 使用数据增强或收集更多样本
2. **调整超参数**: 学习率、批次大小、权重衰减等
3. **更深的网络**: 尝试ResNet50、ResNet101
4. **集成学习**: 训练多个模型进行投票
5. **混合精度训练**: 使用AMP加速训练

## 常见问题

### Q: 训练时显存不足？
A: 减小batch_size或img_size，例如：
```bash
python src/train.py --batch_size 16 --img_size 128
```

### Q: 如何使用CPU训练？
A: 程序会自动检测，如果没有GPU会使用CPU

### Q: 如何继续训练？
A: 可以加载检查点后继续训练（需要修改代码添加resume功能）

### Q: 如何冻结主干网络？
A: 添加 `--freeze_backbone` 参数，只训练分类头

## 示例输出

训练过程：
```
Epoch 1/50:
  Train Loss: 3.4521 | Train Acc: 12.34%
  Val Loss: 3.1234 | Val Acc: 15.67%
  Learning Rate: 0.001000
  Best Val Acc: 15.67% (Epoch 1)
```

预测结果：
```
Top-5 Predictions:
  1. cat: 95.32%
  2. dog: 2.15%
  3. tiger: 1.23%
  4. lion: 0.89%
  5. leopard: 0.41%
```

## 项目贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 作者

Animals90 Recognition System

## 更新日志

- v1.0.0 (2025-12-28): 初始版本发布
  - 实现基础训练、评估、推理功能
  - 支持ResNet18模型
  - 完整的数据增强和可视化
