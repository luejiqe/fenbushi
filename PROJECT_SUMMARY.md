# 项目摘要 - Animals90 动物识别系统

## 项目概述

这是一个基于深度学习的动物图像识别系统，使用PyTorch和ResNet18模型对90种不同的动物进行分类。项目采用迁移学习方法，利用ImageNet预训练权重提高模型性能。

## 核心功能

1. **数据处理** ([src/dataset.py](src/dataset.py))
   - 自定义Animals90Dataset类
   - 完整的数据增强pipeline
   - 高效的DataLoader实现

2. **模型架构** ([src/model.py](src/model.py))
   - 基于ResNet18的迁移学习
   - 自定义分类头（90类输出）
   - 支持冻结/解冻主干网络

3. **训练系统** ([src/train.py](src/train.py))
   - 完整的训练循环
   - 实时验证
   - 自动保存最佳模型
   - 学习率调度（Plateau/Cosine）
   - 训练历史记录

4. **模型评估** ([src/evaluate.py](src/evaluate.py))
   - 测试集性能评估
   - 混淆矩阵可视化
   - 分类报告生成
   - 详细的性能指标

5. **推理预测** ([src/predict.py](src/predict.py))
   - 单张图片推理
   - 批量图片处理
   - Top-K预测结果
   - JSON格式输出

6. **工具函数** ([src/utils.py](src/utils.py))
   - 训练曲线可视化
   - 模型参数统计
   - 性能分析工具

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **计算机视觉**: torchvision
- **数据可视化**: matplotlib, seaborn
- **机器学习**: scikit-learn
- **图像处理**: Pillow

## 文件结构

```
fenbushi/
├── src/                      # 源代码
│   ├── dataset.py           # 数据加载 (153行)
│   ├── model.py             # 模型定义 (89行)
│   ├── train.py             # 训练脚本 (289行)
│   ├── evaluate.py          # 评估脚本 (97行)
│   ├── predict.py           # 推理脚本 (148行)
│   └── utils.py             # 工具函数 (123行)
├── data/                     # 数据目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── checkpoints/             # 模型检查点
├── logs/                    # 训练日志
├── models/                  # 导出模型
├── config.json              # 配置文件
├── quickstart.py            # 快速测试
├── requirements.txt         # 依赖包
├── README.md               # 项目说明
└── USAGE.md                # 使用指南
```

## 关键特性

### 1. 数据增强
- 随机裁剪和翻转
- 随机旋转 (±15°)
- 颜色抖动
- ImageNet归一化

### 2. 训练策略
- Adam优化器
- 交叉熵损失
- 学习率自动调整
- 早停机制
- 最佳模型保存

### 3. 模型优化
- 迁移学习
- Dropout正则化
- 权重衰减
- 可选主干冻结

## 性能指标

预期性能（取决于数据质量和训练配置）：
- **训练时间**: ~2-5小时 (单GPU, 50 epochs)
- **验证准确率**: 85-95% (取决于数据集)
- **推理速度**: ~50-100 images/sec (GPU)
- **模型大小**: ~44MB

## 使用流程

1. **环境准备**
   ```bash
   pip install -r requirements.txt
   python quickstart.py  # 测试环境
   ```

2. **数据准备**
   - 将数据组织到 data/train, data/val, data/test
   - 每个类别一个文件夹

3. **训练模型**
   ```bash
   python src/train.py --train_dir data/train --val_dir data/val
   ```

4. **评估模型**
   ```bash
   python src/evaluate.py --checkpoint checkpoints/checkpoint_best.pth --test_dir data/test
   ```

5. **推理预测**
   ```bash
   python src/predict.py --checkpoint checkpoints/checkpoint_best.pth --image image.jpg
   ```

## 配置选项

主要超参数（可通过命令行修改）：
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批次大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 0.001)
- `--img_size`: 图片大小 (默认: 224)
- `--pretrained`: 使用预训练权重
- `--freeze_backbone`: 冻结主干网络
- `--scheduler`: 学习率调度器 (plateau/cosine)

## 输出文件

训练后生成：
- `checkpoints/checkpoint_best.pth` - 最佳模型
- `checkpoints/checkpoint_latest.pth` - 最新检查点
- `logs/training_history.json` - 训练历史
- `evaluation_results/` - 评估结果目录
  - `classification_report.json`
  - `confusion_matrix.png`

## 代码质量

- ✅ 完整的文档字符串
- ✅ 类型提示
- ✅ 错误处理
- ✅ 进度条显示
- ✅ 日志记录
- ✅ 配置管理

## 扩展性

系统设计考虑了以下扩展：
- 更换其他预训练模型（ResNet50, EfficientNet等）
- 添加更多数据增强方法
- 支持混合精度训练
- 分布式训练
- 模型剪枝和量化
- ONNX导出

## 适用场景

- 动物识别应用
- 计算机视觉教学
- 迁移学习研究
- 图像分类baseline
- 深度学习实践项目

## 开发时间

- 核心代码: ~899行
- 开发时间: 1天
- 测试状态: 已通过基本测试

## 未来改进

- [ ] 添加混合精度训练支持
- [ ] 实现数据集自动划分工具
- [ ] 添加模型可视化工具
- [ ] 支持多GPU训练
- [ ] Web界面demo
- [ ] 移动端部署支持
- [ ] 增加更多预训练模型选项
- [ ] 实现在线数据增强

## 许可证

MIT License - 可自由用于商业和学术项目

---

**最后更新**: 2025-12-28
**项目状态**: 完成 ✓
