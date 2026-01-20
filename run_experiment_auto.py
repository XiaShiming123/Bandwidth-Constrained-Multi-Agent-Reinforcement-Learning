#!/usr/bin/env python3
"""
自动运行实验脚本
处理MPE警告和GPU设置
"""

import os
import sys
import subprocess
import time

# 设置环境变量以抑制MPE警告
os.environ['SUPPRESS_MA_PROMPT'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

print("=" * 60)
print("自动实验运行脚本")
print("=" * 60)

# 检查是否在远程服务器
if sys.platform != 'win32':
    print("检测到Linux环境 (可能是CentOS 7)")
    print(f"工作目录: {os.getcwd()}")

# 导入必要的库
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        gpu_arg = "--gpu 0"
    else:
        print("警告: CUDA不可用，使用CPU模式")
        gpu_arg = "--gpu -1"
        
except ImportError:
    print("错误: PyTorch未安装")
    sys.exit(1)

# 检查MPE
try:
    import multiagent
    print("MPE: 已安装")
except ImportError:
    print("错误: MPE未安装")
    print("请先安装MPE: git clone https://github.com/openai/multiagent-particle-envs.git && cd multiagent-particle-envs && pip install -e .")
    sys.exit(1)

print()

# 实验选项
print("选择实验模式:")
print("1. 快速测试 (验证框架)")
print("2. 小规模实验 (1个算法，1个种子)")
print("3. 完整论文实验 (5个算法，3个种子)")
print("4. 自定义实验")

choice = input("\n请输入选择 (1-4): ").strip()

if choice == "1":
    # 快速测试
    print("\n运行快速测试...")
    
    print("\n1. 测试算法框架...")
    subprocess.run([sys.executable, "test_framework_simple.py"])
    
    print("\n2. 运行示例实验...")
    subprocess.run([sys.executable, "run_example.py"])
    
    print("\n快速测试完成!")
    
elif choice == "2":
    # 小规模实验
    output_dir = f"small_experiment_{int(time.time())}"
    
    print(f"\n运行小规模实验...")
    print(f"输出目录: {output_dir}")
    
    cmd = [
        sys.executable, "experiments/main_experiment.py",
        gpu_arg,
        "--env", "MPE_Navigation",
        "--algorithms", "AdaptiveComm",
        "--seeds", "42",
        "--output_dir", output_dir
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    print(f"\n实验完成! 结果保存在: {output_dir}")
    
elif choice == "3":
    # 完整实验
    output_dir = f"full_experiment_{int(time.time())}"
    
    print(f"\n运行完整论文实验...")
    print(f"输出目录: {output_dir}")
    
    cmd = [
        sys.executable, "experiments/main_experiment.py",
        gpu_arg,
        "--env", "MPE_Navigation",
        "--algorithms", "MAPPO", "IACN", "SparseComm", "FullComm", "AdaptiveComm",
        "--seeds", "42", "123", "456",
        "--output_dir", output_dir
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    print(f"\n实验完成! 结果保存在: {output_dir}")
    
elif choice == "4":
    # 自定义实验
    env = input("\n输入环境 (默认: MPE_Navigation): ") or "MPE_Navigation"
    algorithms = input("输入算法 (用空格分隔，默认: AdaptiveComm MAPPO): ") or "AdaptiveComm MAPPO"
    seeds = input("输入随机种子 (用空格分隔，默认: 42 123): ") or "42 123"
    
    # 解析GPU参数
    if "--gpu" in gpu_arg:
        gpu_id = gpu_arg.split()[-1]
    else:
        gpu_id = "-1"
    
    custom_gpu = input(f"GPU ID (默认: {gpu_id}): ") or gpu_id
    
    output_dir = f"custom_experiment_{int(time.time())}"
    
    print(f"\n实验配置:")
    print(f"  环境: {env}")
    print(f"  算法: {algorithms}")
    print(f"  种子: {seeds}")
    print(f"  GPU: {custom_gpu}")
    print(f"  输出目录: {output_dir}")
    
    confirm = input("\n是否开始实验? (y/n): ").strip().lower()
    if confirm != 'y':
        print("用户取消")
        sys.exit(0)
    
    # 构建命令
    cmd = [sys.executable, "experiments/main_experiment.py"]
    cmd.extend(["--gpu", custom_gpu])
    cmd.extend(["--env", env])
    cmd.extend(["--algorithms"] + algorithms.split())
    cmd.extend(["--seeds"] + seeds.split())
    cmd.extend(["--output_dir", output_dir])
    
    print(f"\n执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    print(f"\n实验完成! 结果保存在: {output_dir}")
    
else:
    print("无效选择")
    sys.exit(1)

print("\n" + "=" * 60)
print("脚本执行完成")
print("=" * 60)