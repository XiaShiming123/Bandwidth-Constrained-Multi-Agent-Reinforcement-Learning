#!/usr/bin/env python3
"""
检查环境脚本
"""

import os
import sys
import torch

print("=" * 60)
print("环境检查")
print("=" * 60)

# 基本信息
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.1f}GB)")
else:
    print("警告: CUDA不可用，将使用CPU")

# 检查MPE
print("\n检查MPE:")
try:
    import multiagent
    print("  MPE: 已安装")
except:
    print("  MPE: 未安装")

print("\n" + "=" * 60)
print("运行实验命令:")
print("=" * 60)

if torch.cuda.is_available():
    print("python experiments/main_experiment.py --gpu 0 --env MPE_Navigation --algorithms AdaptiveComm MAPPO")
else:
    print("python experiments/main_experiment.py --gpu -1 --env MPE_Navigation --algorithms AdaptiveComm MAPPO")
    print("\n注意: 使用CPU模式，训练会较慢")

print("\n测试命令:")
print("python test_framework_simple.py")
print("python run_example.py")