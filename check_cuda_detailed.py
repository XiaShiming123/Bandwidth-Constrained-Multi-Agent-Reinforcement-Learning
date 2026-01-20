#!/usr/bin/env python3
"""
详细检查CUDA和GPU状态
"""

import os
import sys
import subprocess
import torch

print("=" * 80)
print("CUDA/GPU详细检查")
print("=" * 80)

# 1. 检查PyTorch CUDA支持
print("\n1. PyTorch CUDA支持:")
print(f"   PyTorch版本: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")

if hasattr(torch.version, 'cuda'):
    print(f"   PyTorch编译的CUDA版本: {torch.version.cuda}")
else:
    print("   PyTorch未编译CUDA支持")

# 2. 检查系统CUDA
print("\n2. 系统CUDA检查:")
try:
    # 检查nvcc
    result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   nvcc路径: {result.stdout.strip()}")
        
        # 检查CUDA版本
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"   系统CUDA版本: {line.strip()}")
    else:
        print("   nvcc未找到")
except Exception as e:
    print(f"   检查nvcc失败: {e}")

# 3. 检查NVIDIA驱动
print("\n3. NVIDIA驱动检查:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   NVIDIA驱动信息:")
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines[:10]):  # 只显示前10行
            print(f"     {line}")
    else:
        print("   nvidia-smi未找到或失败")
except Exception as e:
    print(f"   检查NVIDIA驱动失败: {e}")

# 4. 检查GPU设备
print("\n4. GPU设备检查:")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"   GPU数量: {device_count}")
    
    for i in range(device_count):
        print(f"   GPU {i}:")
        print(f"     名称: {torch.cuda.get_device_name(i)}")
        print(f"     内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"     CUDA计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("   无可用GPU设备")

# 5. 检查CUDA兼容性
print("\n5. CUDA兼容性检查:")
if hasattr(torch.version, 'cuda') and torch.cuda.is_available():
    pytorch_cuda_version = torch.version.cuda
    
    # 尝试获取系统CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            if 'release' in output:
                import re
                match = re.search(r'release (\d+\.\d+)', output)
                if match:
                    system_cuda_version = match.group(1)
                    print(f"   PyTorch CUDA版本: {pytorch_cuda_version}")
                    print(f"   系统CUDA版本: {system_cuda_version}")
                    
                    # 检查版本兼容性
                    pytorch_major = int(pytorch_cuda_version.split('.')[0])
                    system_major = int(system_cuda_version.split('.')[0])
                    
                    if pytorch_major == system_major:
                        print("   ✅ CUDA版本兼容")
                    else:
                        print(f"   ⚠️ CUDA版本不兼容: PyTorch需要CUDA {pytorch_major}.x，系统是CUDA {system_major}.x")
    except:
        print("   无法检查系统CUDA版本")

# 6. 测试CUDA功能
print("\n6. CUDA功能测试:")
if torch.cuda.is_available():
    try:
        # 测试张量移动
        x = torch.randn(3, 3).cuda()
        print(f"   ✅ 张量移动到GPU成功: {x.device}")
        
        # 测试GPU计算
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print(f"   ✅ GPU计算成功: {z.shape}")
        
        # 测试内存分配
        large_tensor = torch.randn(1000, 1000).cuda()
        print(f"   ✅ 大内存分配成功: {large_tensor.nelement() * 4 / 1024**2:.2f} MB")
        
        del x, y, z, large_tensor
        torch.cuda.empty_cache()
        print("   ✅ GPU内存清理成功")
        
    except Exception as e:
        print(f"   ❌ CUDA测试失败: {e}")
else:
    print("   CUDA不可用，跳过测试")

# 7. 解决方案建议
print("\n7. 解决方案建议:")
print("=" * 40)

if not torch.cuda.is_available():
    print("\n问题: CUDA不可用")
    print("\n可能的原因和解决方案:")
    print("""
1. 没有NVIDIA GPU
   - 检查服务器是否有NVIDIA GPU: lspci | grep -i nvidia
   - 如果没有GPU，考虑使用云GPU服务

2. NVIDIA驱动未安装
   - 安装驱动: sudo yum install nvidia-driver-latest-dkms (CentOS)
   - 或从NVIDIA官网下载驱动

3. CUDA工具包未安装
   - 安装CUDA: https://developer.nvidia.com/cuda-downloads
   - CentOS 7: sudo yum install cuda

4. PyTorch CUDA版本不匹配
   - 重新安装匹配的PyTorch:
     pip uninstall torch torchvision
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

5. 环境变量问题
   - 检查LD_LIBRARY_PATH: echo $LD_LIBRARY_PATH
   - 添加CUDA库路径: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

6. 权限问题
   - 检查用户是否有GPU访问权限
   - 将用户添加到video组: sudo usermod -a -G video $USER
""")
else:
    print("✅ CUDA可用，可以正常使用GPU")
    print("\n使用GPU运行实验的命令:")
    print("python experiments/main_experiment.py --gpu 0 --env MPE_Navigation --algorithms AdaptiveComm")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)