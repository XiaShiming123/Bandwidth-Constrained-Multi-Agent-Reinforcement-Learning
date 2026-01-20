#!/usr/bin/env python3
"""
简单安装MPE环境
"""

import os
import sys
import subprocess

print("Installing MPE environment...")

# 1. 克隆MPE仓库
print("\n1. Cloning MPE repository...")
if os.path.exists("multiagent-particle-envs"):
    print("   MPE repository already exists")
else:
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/openai/multiagent-particle-envs.git"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   Successfully cloned MPE repository")
        else:
            print(f"   Failed to clone: {result.stderr}")
            sys.exit(1)
    except Exception as e:
        print(f"   Error cloning: {e}")
        sys.exit(1)

# 2. 安装MPE
print("\n2. Installing MPE...")
try:
    original_dir = os.getcwd()
    os.chdir("multiagent-particle-envs")
    
    # 安装
    result = subprocess.run(
        [sys.executable, "setup.py", "develop"],
        capture_output=True,
        text=True
    )
    
    os.chdir(original_dir)
    
    if result.returncode == 0:
        print("   Successfully installed MPE")
    else:
        print(f"   Failed to install: {result.stderr}")
        sys.exit(1)
        
except Exception as e:
    print(f"   Error installing: {e}")
    sys.exit(1)

# 3. 验证安装
print("\n3. Verifying installation...")
try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("   MPE successfully imported")
    
    # 测试场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"   Available scenarios: {len(scenario_names)}")
    
    print("\n" + "=" * 60)
    print("MPE installation completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"   Verification failed: {e}")
    sys.exit(1)