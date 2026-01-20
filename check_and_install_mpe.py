#!/usr/bin/env python3
"""
检查并安装MPE环境
"""

import sys
import os

print("检查MPE环境...")

# 尝试导入MPE
try:
    import multiagent
    print("✓ MPE已安装")
    
    # 测试环境创建
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("✓ MPE环境可以正常创建")
    
    # 列出可用的场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"可用场景: {scenario_names[:10]}..." if len(scenario_names) > 10 else f"可用场景: {scenario_names}")
    
    sys.exit(0)
    
except ImportError as e:
    print(f"✗ MPE未安装: {e}")
    print("\n需要安装MPE环境:")
    print("1. git clone https://github.com/openai/multiagent-particle-envs.git")
    print("2. cd multiagent-particle-envs")
    print("3. pip install -e .")
    print("\n或者运行安装脚本:")
    print("python install_mpe_remote.py")
    
    # 询问是否自动安装
    response = input("\n是否自动安装MPE? (y/n): ")
    if response.lower() == 'y':
        print("\n开始安装MPE...")
        
        # 克隆仓库
        if not os.path.exists("multiagent-particle-envs"):
            os.system("git clone https://github.com/openai/multiagent-particle-envs.git")
        
        # 安装
        os.chdir("multiagent-particle-envs")
        os.system(f"{sys.executable} setup.py develop")
        os.chdir("..")
        
        print("\n安装完成! 请重新运行此脚本验证安装。")
    
    sys.exit(1)

except Exception as e:
    print(f"✗ MPE导入错误: {e}")
    sys.exit(1)