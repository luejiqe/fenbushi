"""
PyTorch MUSA Compatibility Patch
修复PyTorch multiprocessing中的MUSA检测问题
"""

import torch.multiprocessing.reductions as reductions
import torch

# 保存原始的reduce_storage函数
original_reduce_storage = reductions.reduce_storage


def patched_reduce_storage(storage):
    """
    修补后的reduce_storage函数，添加is_musa属性检查
    """
    # 如果storage没有is_musa属性，添加一个返回False的属性
    if not hasattr(storage, 'is_musa'):
        storage.is_musa = False

    return original_reduce_storage(storage)


# 应用补丁
reductions.reduce_storage = patched_reduce_storage

print("✓ PyTorch MUSA compatibility patch applied successfully")
print("  Now you can use num_workers > 0 without errors")
