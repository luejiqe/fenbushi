# 摩尔线程GPU支持说明

## ✅ 已完成的修改

本项目现已支持**自动检测**并使用以下设备：

1. **NVIDIA GPU (CUDA)** - 优先级最高
2. **摩尔线程GPU (MUSA)** - 次优先级
3. **CPU** - 备选方案

## 📝 修改的文件

已修改以下文件以支持摩尔线程GPU（MUSA）：

### 1. [src/train.py](src/train.py)
- 自动检测CUDA、MUSA或CPU
- 训练时会显示使用的设备类型和GPU名称

### 2. [src/model.py](src/model.py)
- `create_model()` 函数支持MUSA设备
- 自动设备选择逻辑

### 3. [src/predict.py](src/predict.py)
- 推理时自动检测并使用MUSA
- `Animals90Predictor` 类支持MUSA

### 4. [src/evaluate.py](src/evaluate.py)
- 评估时自动检测并使用MUSA
- 命令行参数现在支持 `--device musa`

### 5. [quickstart.py](quickstart.py)
- 环境检测会显示MUSA信息
- 显示MUSA设备名称和数量

## 🚀 使用方法

### 无需添加任何参数！

代码会**自动检测**可用的GPU类型：

```bash
# 训练（自动检测）
python src/train.py --train_dir data/train --val_dir data/val

# 预测（自动检测）
python src/predict.py --checkpoint checkpoints/checkpoint_best.pth --image test.jpg

# 评估（自动检测）
python src/evaluate.py --checkpoint checkpoints/checkpoint_best.pth --test_dir data/test
```

### 设备检测优先级

1. 首先检查 CUDA (NVIDIA GPU)
2. 如果没有CUDA，检查 MUSA (摩尔线程GPU)
3. 如果都没有，使用 CPU

## 📊 预期输出

### 使用NVIDIA GPU时：
```
Using device: cuda
GPU: NVIDIA GeForce MX450
```

### 使用摩尔线程GPU时：
```
Using device: musa
GPU: Moore Threads GPU
```

### 使用CPU时：
```
Using device: cpu
```

## 🔧 环境要求

### 使用摩尔线程GPU需要：

1. **安装torch_musa包**
   ```bash
   # 参考摩尔线程官方文档安装
   pip install torch_musa
   ```

2. **摩尔线程驱动**
   - 确保已安装摩尔线程GPU驱动
   - 验证GPU可用性

### 验证环境

运行快速测试：
```bash
python quickstart.py
```

输出示例（摩尔线程GPU）：
```
Environment Check
============================================================
Python version: 3.9.x
PyTorch version: 2.x.x
CUDA available: False
MUSA available: True
MUSA device: Moore Threads MTT S80
MUSA count: 1

Testing Model Creation and Forward Pass
============================================================
1. Creating model...
✓ Model created successfully on musa
```

## 💡 技术细节

### 设备检测代码逻辑

```python
# 自动检测设备
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    try:
        import torch_musa
        if torch_musa.is_available():
            device = torch.device('musa')
        else:
            device = torch.device('cpu')
    except ImportError:
        device = torch.device('cpu')
```

### 特点

✅ **零配置** - 无需修改命令或添加参数
✅ **自动检测** - 智能选择最佳可用设备
✅ **向后兼容** - 在没有MUSA的环境中正常工作
✅ **优雅降级** - CUDA > MUSA > CPU

## ⚠️ 注意事项

1. **torch_musa包是可选的**
   - 如果未安装，代码会自动降级到CPU
   - 不会报错或中断

2. **设备优先级**
   - NVIDIA GPU (CUDA) 优先级最高
   - 如果同时有CUDA和MUSA，会优先使用CUDA

3. **性能对比**
   - MUSA性能取决于具体的摩尔线程GPU型号
   - 建议测试实际训练速度进行对比

## 🎯 常见问题

### Q: 如何强制使用CPU？
A: 目前代码会自动选择最佳设备。如需强制CPU，可以临时禁用GPU。

### Q: 如何验证正在使用MUSA？
A: 运行训练时，开头会显示 `Using device: musa`

### Q: MUSA训练速度如何？
A: 取决于具体的摩尔线程GPU型号，通常比CPU快5-20倍。

### Q: 是否需要修改现有命令？
A: 不需要！所有现有命令保持不变，代码会自动检测。

## 📚 参考资料

- 摩尔线程官方文档
- PyTorch MUSA后端文档
- torch_musa安装指南

---

**最后更新**: 2025-12-28
**状态**: ✅ 已完成并测试
