#!/usr/bin/env python3
"""
在远程服务器上安装MPE环境的脚本

通过PyCharm远程执行（Ramgmal解释器）运行
"""

import os
import sys
import subprocess
import platform
import time

print("=" * 80)
print("在远程服务器上安装MPE环境")
print("=" * 80)
print()

# 显示环境信息
print("环境信息:")
print(f"  工作目录: {os.getcwd()}")
print(f"  Python版本: {sys.version}")
print(f"  Python路径: {sys.executable}")
print(f"  操作系统: {platform.system()} {platform.release()}")
print()

# 步骤1: 检查是否已安装MPE
print("步骤1: 检查MPE是否已安装")
print("-" * 40)

try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("✓ MPE已安装")
    print(f"  可用场景: {[name for name in dir(scenarios) if not name.startswith('_')][:5]}...")
    
    # 测试环境创建
    print("  测试环境创建...")
    scenario = scenarios.load("simple.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    print("  ✓ MPE环境创建成功")
    
    print("\nMPE环境已就绪，可以开始实验!")
    sys.exit(0)
    
except ImportError:
    print("✗ MPE未安装")
    print("  开始安装过程...")
except Exception as e:
    print(f"✗ MPE导入错误: {e}")
    print("  需要重新安装或修复MPE...")

print()

# 步骤2: 安装git（如果需要）
print("步骤2: 检查git是否可用")
print("-" * 40)

try:
    result = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Git可用: {result.stdout.strip()}")
    else:
        print("✗ Git不可用，尝试安装...")
        # 这里可以添加git安装逻辑
        print("  请手动安装git: sudo apt-get install git")
        sys.exit(1)
except FileNotFoundError:
    print("✗ Git未安装")
    print("  请手动安装git: sudo apt-get install git")
    sys.exit(1)

print()

# 步骤3: 克隆MPE仓库
print("步骤3: 克隆MPE仓库")
print("-" * 40)

mpe_dir = "multiagent-particle-envs"
if os.path.exists(mpe_dir):
    print(f"✓ MPE目录已存在: {mpe_dir}")
    print("  跳过克隆步骤")
else:
    print("  克隆MPE仓库...")
    
    # 尝试多个镜像源
    repos = [
        "https://github.com/openai/multiagent-particle-envs.git",
        "https://gitee.com/mirrors/multiagent-particle-envs.git",
        "git@github.com:openai/multiagent-particle-envs.git"
    ]
    
    success = False
    for repo_url in repos:
        print(f"  尝试从 {repo_url} 克隆...")
        try:
            result = subprocess.run(
                ["git", "clone", repo_url, mpe_dir],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✓ 克隆成功")
                success = True
                break
            else:
                print(f"  ✗ 克隆失败: {result.stderr[:100]}...")
                
        except subprocess.TimeoutExpired:
            print("  ✗ 克隆超时")
        except Exception as e:
            print(f"  ✗ 克隆异常: {e}")
    
    if not success:
        print("\n所有克隆源都失败，请检查网络连接或手动下载MPE")
        print("手动下载链接: https://github.com/openai/multiagent-particle-envs")
        sys.exit(1)

print()

# 步骤4: 安装MPE
print("步骤4: 安装MPE")
print("-" * 40)

print("  进入MPE目录...")
original_dir = os.getcwd()
try:
    os.chdir(mpe_dir)
    print(f"  当前目录: {os.getcwd()}")
    
    # 检查setup.py是否存在
    if not os.path.exists("setup.py"):
        print("  ✗ setup.py不存在，MPE仓库可能不完整")
        sys.exit(1)
    
    print("  运行安装命令...")
    
    # 尝试多种安装方式
    install_commands = [
        [sys.executable, "setup.py", "develop"],
        [sys.executable, "-m", "pip", "install", "-e", "."],
        ["pip", "install", "-e", "."]
    ]
    
    install_success = False
    for cmd in install_commands:
        print(f"  尝试: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("  ✓ 安装成功")
                install_success = True
                break
            else:
                print(f"  ✗ 安装失败: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            print("  ✗ 安装超时")
        except Exception as e:
            print(f"  ✗ 安装异常: {e}")
    
    if not install_success:
        print("\n所有安装方式都失败")
        sys.exit(1)
    
finally:
    # 返回原目录
    os.chdir(original_dir)

print()

# 步骤5: 验证安装
print("步骤5: 验证MPE安装")
print("-" * 40)

print("  等待安装完成...")
time.sleep(2)  # 给系统一些时间更新模块路径

print("  重新导入MPE...")
# 重新加载sys.path以确保找到新安装的模块
import site
import importlib
importlib.invalidate_caches()

# 添加MPE目录到Python路径
mpe_path = os.path.join(original_dir, mpe_dir)
sys.path.insert(0, mpe_path)

success = False
try:
    # 清除可能的旧模块
    if 'multiagent' in sys.modules:
        del sys.modules['multiagent']
    
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("  ✓ MPE导入成功")
    
    # 列出可用场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"  可用场景数量: {len(scenario_names)}")
    print(f"  示例场景: {scenario_names[:5]}")
    
    success = True
    
except ImportError as e:
    print(f"  ✗ MPE导入失败: {e}")
    print("  可能的原因:")
    print("  1. 安装未完全成功")
    print("  2. Python路径问题")
    print("  3. 依赖包缺失")
    
except Exception as e:
    print(f"  ✗ 验证过程中出错: {e}")

print()

# 最终结果
print("=" * 80)
if success:
    print("✅ MPE环境安装成功!")
    print()
    print("下一步:")
    print("1. 运行测试: python test_framework_simple.py")
    print("2. 运行实验: python experiments/main_experiment.py")
    print("3. 查看示例: python run_example.py")
else:
    print("❌ MPE环境安装失败")
    print()
    print("请尝试:")
    print("1. 手动安装: cd multiagent-particle-envs && pip install -e .")
    print("2. 检查网络连接")
    print("3. 查看详细错误信息")

print("=" * 80)