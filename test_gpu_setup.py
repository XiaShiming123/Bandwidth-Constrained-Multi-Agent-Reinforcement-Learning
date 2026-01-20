#!/usr/bin/env python3
"""
测试GPU设置
"""

import torch
import sys

print("=" * 60)
print("GPU设置测试")
print("=" * 60)

# 检查CUDA是否可用
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # 测试张量移动到GPU
    print("\n测试张量移动:")
    x = torch.randn(3, 3)
    print(f"  CPU张量: {x.device}")
    
    x_gpu = x.cuda()
    print(f"  GPU张量: {x_gpu.device}")
    
    # 测试计算
    y = torch.randn(3, 3).cuda()
    z = torch.matmul(x_gpu, y)
    print(f"  GPU计算成功: {z.shape}")
    
    # 测试不同GPU
    if torch.cuda.device_count() > 1:
        print("\n测试多GPU:")
        for i in range(min(2, torch.cuda.device_count())):
            torch.cuda.set_device(i)
            x = torch.randn(3, 3).cuda()
            print(f"  GPU {i}: 张量设备 {x.device}")
    
    print("\n✅ GPU设置正常")
    
else:
    print("\n❌ CUDA不可用")
    print("可能的原因:")
    print("1. 未安装CUDA版本的PyTorch")
    print("2. 没有NVIDIA GPU")
    print("3. CUDA驱动未安装")
    print("4. PyTorch版本与CUDA版本不匹配")
    
    # 检查PyTorch构建配置
    print(f"\nPyTorch构建配置:")
    print(f"  使用MKL: {torch.backends.mkl.is_available()}")
    print(f"  使用OpenMP: {torch.backends.openmp.is_available()}")
    print(f"  CUDA后端: {torch.backends.cudnn.enabled}")

print("\n" + "=" * 60)