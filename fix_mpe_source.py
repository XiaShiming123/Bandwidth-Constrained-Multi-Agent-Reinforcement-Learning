#!/usr/bin/env python3
"""
直接修复MPE源代码中的Gym兼容性问题
"""

import os
import sys

print("=" * 80)
print("修复MPE源代码")
print("=" * 80)

# MPE目录
mpe_dir = "multiagent-particle-envs"
if not os.path.exists(mpe_dir):
    print(f"❌ MPE目录不存在: {mpe_dir}")
    print("请先克隆MPE仓库: git clone https://github.com/openai/multiagent-particle-envs.git")
    sys.exit(1)

# 修复文件1: multi_discrete.py
print("\n1. 修复 multi_discrete.py...")
multi_discrete_file = os.path.join(mpe_dir, "multiagent", "multi_discrete.py")

if os.path.exists(multi_discrete_file):
    with open(multi_discrete_file, 'r') as f:
        content = f.read()
    
    # 备份原文件
    backup_file = multi_discrete_file + ".backup"
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"  备份到: {backup_file}")
    
    # 修复导入
    if "from gym.spaces import prng" in content:
        content = content.replace(
            "from gym.spaces import prng",
            "# from gym.spaces import prng  # 已弃用\nimport numpy as np"
        )
        print("  修复了导入语句")
    
    # 修复prng使用
    if "self.np_random = prng.np_random" in content:
        content = content.replace(
            "self.np_random = prng.np_random",
            "self.np_random = np.random.RandomState()"
        )
        print("  修复了prng.np_random使用")
    
    # 写回文件
    with open(multi_discrete_file, 'w') as f:
        f.write(content)
    
    print(f"  ✅ {multi_discrete_file} 已修复")
else:
    print(f"  ❌ 文件不存在: {multi_discrete_file}")

# 修复文件2: __init__.py (移除警告)
print("\n2. 修复 __init__.py...")
init_file = os.path.join(mpe_dir, "multiagent", "__init__.py")

if os.path.exists(init_file):
    with open(init_file, 'r') as f:
        lines = f.readlines()
    
    # 备份
    backup_file = init_file + ".backup"
    with open(backup_file, 'w') as f:
        f.writelines(lines)
    print(f"  备份到: {backup_file}")
    
    # 移除弃用警告
    new_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
            
        if 'warnings.warn' in line and 'This code base is no longer maintained' in line:
            # 跳过警告行
            print("  移除了弃用警告")
            # 检查下一行是否是续行
            if i + 1 < len(lines) and lines[i+1].strip().endswith(')'):
                skip_next = True
            continue
        
        new_lines.append(line)
    
    # 写回文件
    with open(init_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"  ✅ {init_file} 已修复")
else:
    print(f"  ❌ 文件不存在: {init_file}")

# 修复文件3: environment.py (如果需要)
print("\n3. 检查 environment.py...")
env_file = os.path.join(mpe_dir, "multiagent", "environment.py")

if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        content = f.read()
    
    # 检查是否有gym.spaces.prng导入
    if "from gym.spaces import prng" in content:
        print("  ⚠️  environment.py 可能也需要修复")
        print("  但通常只需要修复multi_discrete.py")
    else:
        print("  ✅ environment.py 看起来正常")
else:
    print(f"  ❌ 文件不存在: {env_file}")

# 重新安装MPE
print("\n4. 重新安装MPE...")
original_dir = os.getcwd()
try:
    os.chdir(mpe_dir)
    print(f"  当前目录: {os.getcwd()}")
    
    # 检查setup.py
    if os.path.exists("setup.py"):
        import subprocess
        result = subprocess.run([sys.executable, "setup.py", "develop"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ MPE重新安装成功")
        else:
            print(f"  ❌ 重新安装失败: {result.stderr[:200]}")
    else:
        print("  ❌ setup.py不存在")
    
finally:
    os.chdir(original_dir)

# 测试修复
print("\n5. 测试修复...")
try:
    # 设置环境变量
    os.environ['SUPPRESS_MA_PROMPT'] = '1'
    
    # 导入兼容层
    import gym_compat
    
    # 导入MPE
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("  ✅ MPE导入成功")
    
    # 列出场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"  可用场景: {len(scenario_names)}个")
    
    print("\n  ✅ 所有修复成功!")
    
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    print("\n  可能需要:")
    print("    1. 重启Python环境")
    print("    2. 手动重新安装MPE: cd multiagent-particle-envs && pip install -e .")

print("\n" + "=" * 80)
print("修复完成")
print("=" * 80)