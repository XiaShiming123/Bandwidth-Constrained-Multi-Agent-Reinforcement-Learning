#!/usr/bin/env python3
"""
快速修复Gym/MPE兼容性问题
"""

import os
import sys

print("=" * 60)
print("快速修复Gym/MPE兼容性问题")
print("=" * 60)

# 1. 安装Gymnasium
print("\n1. 安装Gymnasium...")
os.system(f"{sys.executable} -m pip install gymnasium -q")

# 2. 创建猴子补丁
print("\n2. 创建猴子补丁...")

monkey_patch_code = '''
"""
Gym猴子补丁 - 解决MPE兼容性问题
在导入任何其他模块之前导入此文件
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# 导入gymnasium并替换gym
import gymnasium as gym
sys.modules['gym'] = gym

# 为gym.spaces添加prng属性
import numpy as np

class FakePRNG:
    def __init__(self):
        self.np_random = np.random.RandomState()

if not hasattr(gym.spaces, 'prng'):
    gym.spaces.prng = FakePRNG()

if not hasattr(gym.spaces, 'np_random'):
    gym.spaces.np_random = np.random.RandomState()

print("✅ Gym猴子补丁已应用: gym -> gymnasium")
'''

with open("gym_monkey_patch.py", "w") as f:
    f.write(monkey_patch_code)

print("  ✅ 创建了 gym_monkey_patch.py")

# 3. 修改环境文件
print("\n3. 修改环境文件...")
env_file = "experiments/environments/multiagent_env.py"

if os.path.exists(env_file):
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # 在文件开头添加导入
    new_lines = []
    added = False
    
    for line in lines:
        # 在import语句之前添加猴子补丁
        if not added and line.strip().startswith("import"):
            new_lines.append("# 应用Gym猴子补丁解决兼容性问题\n")
            new_lines.append("import gym_monkey_patch\n")
            new_lines.append("\n")
            added = True
        new_lines.append(line)
    
    with open(env_file, "w") as f:
        f.writelines(new_lines)
    
    print(f"  ✅ 修改了 {env_file}")
else:
    print(f"  ⚠️  文件不存在: {env_file}")

# 4. 测试修复
print("\n4. 测试修复...")

test_code = '''
# 应用猴子补丁
import gym_monkey_patch

# 现在应该可以导入MPE了
try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("✅ MPE导入成功!")
    
    # 测试场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"可用场景: {len(scenario_names)}个")
    
    # 测试环境创建
    scenario = scenarios.load("simple.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    print("✅ MPE环境创建成功!")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
'''

with open("test_fix.py", "w") as f:
    f.write(test_code)

os.system(f"{sys.executable} test_fix.py")

# 5. 清理
print("\n5. 清理临时文件...")
if os.path.exists("test_fix.py"):
    os.remove("test_fix.py")
    print("  ✅ 删除了 test_fix.py")

print("\n" + "=" * 60)
print("修复完成!")
print("=" * 60)
print("\n现在可以运行实验:")
print("python experiments/main_experiment_fixed.py --gpu 0 --env MPE_Navigation --algorithms AdaptiveComm")
print("\n或者运行完整实验:")
print("python experiments/main_experiment_fixed.py --gpu 0 --env MPE_Navigation --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm --seeds 42 123 456")