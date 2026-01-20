#!/usr/bin/env python3
"""
Gym补丁模块 - 解决MPE兼容性问题
"""

import sys
import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')

# 设置环境变量
import os
os.environ['SUPPRESS_MA_PROMPT'] = '1'

print("=" * 60)
print("应用Gym补丁")
print("=" * 60)

# 尝试导入gymnasium
print("1. 导入Gymnasium...")
try:
    import gymnasium as gym
    print(f"   ✅ Gymnasium版本: {gym.__version__}")
    
    # 将gym模块替换为gymnasium
    sys.modules['gym'] = gym
    print("   ✅ 已将gym替换为gymnasium")
    
except ImportError:
    print("   ⚠️  Gymnasium未安装，尝试导入gym")
    try:
        import gym
        print(f"   ✅ Gym版本: {gym.__version__}")
    except ImportError:
        print("   ❌ Gym也未安装")
        sys.exit(1)

# 为旧版MPE添加缺失的属性
print("\n2. 添加缺失的属性...")
try:
    import numpy as np
    
    # 创建伪prng类
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    # 确保gym.spaces有prng属性
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
        print("   ✅ 添加了 gym.spaces.prng")
    
    # 确保有np_random属性
    if not hasattr(gym.spaces, 'np_random'):
        gym.spaces.np_random = np.random.RandomState()
        print("   ✅ 添加了 gym.spaces.np_random")
    
    # 打印状态
    print(f"   gym.spaces类型: {type(gym.spaces)}")
    print(f"   是否有prng属性: {hasattr(gym.spaces, 'prng')}")
    print(f"   是否有np_random属性: {hasattr(gym.spaces, 'np_random')}")
    
except Exception as e:
    print(f"   ⚠️  添加属性时出错: {e}")

# 测试MPE导入
print("\n3. 测试MPE导入...")
try:
    import multiagent
    print("   ✅ MPE导入成功")
    
    # 检查版本
    print(f"   MPE路径: {multiagent.__file__}")
    
except ImportError as e:
    print(f"   ❌ MPE导入失败: {e}")
    print("   可能需要重新安装MPE: cd multiagent-particle-envs && pip install -e .")
except Exception as e:
    print(f"   ⚠️  MPE导入警告: {e}")

print("\n" + "=" * 60)
print("Gym补丁应用完成!")
print("=" * 60)

# 如果直接运行此脚本，进行完整测试
if __name__ == "__main__":
    print("\n完整测试:")
    try:
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
        
        print("   ✅ MultiAgentEnv导入成功")
        print("   ✅ scenarios导入成功")
        
        # 列出场景
        scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
        print(f"   可用场景: {len(scenario_names)}个")
        if scenario_names:
            print(f"   示例: {scenario_names[:3]}")
        
        print("\n✅ 所有测试通过!")
        
    except Exception as e:
        print(f"   ❌ 详细测试失败: {e}")
        import traceback
        traceback.print_exc()