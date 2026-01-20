#!/usr/bin/env python3
"""
在远程服务器上安装MPE环境的脚本

通过PyCharm的远程执行机制运行
"""

import os
import sys
import subprocess
import platform

print("=" * 60)
print("安装MPE (Multi-Agent Particle Environment)")
print("=" * 60)

# 检查当前环境
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")
print(f"操作系统: {platform.system()} {platform.release()}")

# 检查是否已经安装了MPE
try:
    import multiagent
    print("\n✓ MPE已经安装")
    print(f"MPE版本: {multiagent.__version__ if hasattr(multiagent, '__version__') else '未知'}")
    sys.exit(0)
except ImportError:
    print("\n✗ MPE未安装，开始安装...")

# 安装步骤
print("\n1. 克隆MPE仓库...")
try:
    # 检查是否已经克隆
    if os.path.exists("multiagent-particle-envs"):
        print("   MPE仓库已存在，跳过克隆")
    else:
        # 克隆仓库
        result = subprocess.run(
            ["git", "clone", "https://github.com/openai/multiagent-particle-envs.git"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("   ✓ MPE仓库克隆成功")
        else:
            print(f"   ✗ 克隆失败: {result.stderr}")
            sys.exit(1)
except Exception as e:
    print(f"   ✗ 克隆异常: {e}")
    sys.exit(1)

print("\n2. 安装MPE...")
try:
    mpe_dir = os.path.join(os.getcwd(), "multiagent-particle-envs")
    
    # 进入MPE目录
    original_dir = os.getcwd()
    os.chdir(mpe_dir)
    
    # 安装MPE
    result = subprocess.run(
        [sys.executable, "setup.py", "develop"],
        capture_output=True,
        text=True
    )
    
    # 返回原目录
    os.chdir(original_dir)
    
    if result.returncode == 0:
        print("   ✓ MPE安装成功")
    else:
        print(f"   ✗ 安装失败: {result.stderr}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ 安装异常: {e}")
    sys.exit(1)

print("\n3. 验证安装...")
try:
    import multiagent
    print("   ✓ MPE导入成功")
    
    # 测试是否可以创建环境
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("   ✓ MPE组件导入成功")
    print("\n" + "=" * 60)
    print("MPE环境安装完成!")
    print("=" * 60)
    print("\n现在可以运行实验:")
    print("python experiments/main_experiment.py --env MPE_Navigation --algorithms AdaptiveComm")
    
except Exception as e:
    print(f"   ✗ 验证失败: {e}")
    sys.exit(1)