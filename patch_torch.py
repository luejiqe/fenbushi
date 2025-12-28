"""
Permanent MUSA Compatibility Fix
永久修复PyTorch的MUSA兼容性问题
在导入任何torch模块之前运行此脚本
"""

def fix_torch_musa_compatibility():
    """修复torch.storage缺少is_musa属性的问题"""
    import torch

    # 获取所有storage类型
    storage_classes = [
        torch.storage.UntypedStorage,
        torch.FloatStorage,
        torch.DoubleStorage,
        torch.HalfStorage,
        torch.BFloat16Storage,
        torch.LongStorage,
        torch.IntStorage,
        torch.ShortStorage,
        torch.CharStorage,
        torch.ByteStorage,
        torch.BoolStorage,
    ]

    # 为每个storage类添加is_musa属性
    for storage_class in storage_classes:
        if not hasattr(storage_class, 'is_musa'):
            # 添加is_musa属性，始终返回False
            storage_class.is_musa = property(lambda self: False)

    print("✓ PyTorch MUSA compatibility fix applied to all storage classes")

# 立即应用修复
fix_torch_musa_compatibility()
