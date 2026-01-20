#!/usr/bin/env python3
"""
Gym兼容层 - 将Gym调用重定向到Gymnasium
解决MPE与新版Gym的兼容性问题
"""

import sys
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 尝试导入gymnasium
import gymnasium as gym

# 将gym模块替换为gymnasium
sys.modules['gym'] = gym

# 为旧版MPE添加缺失的属性
if hasattr(gym, 'spaces'):
    import numpy as np
    
    # 创建伪prng模块
    class PRNGWrapper:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    # 添加到gym.spaces
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = PRNGWrapper()
    
    # 确保有np_random属性
    if not hasattr(gym.spaces, 'np_random'):
        gym.spaces.np_random = np.random.RandomState()

# 复制其他必要属性
if hasattr(gym, 'Env'):
    # 确保有Env类
    pass

# 打印状态
print("✅ Gym兼容层已激活: 使用Gymnasium替代Gym")
print(f"   版本: {gym.__version__}")

# 测试导入
if __name__ == "__main__":
    print("\n测试导入:")
    try:
        # 测试gym导入
        import gym as test_gym
        print(f"  gym模块: {test_gym.__file__}")
        
        # 测试spaces
        from gym import spaces
        print(f"  spaces模块: {spaces.__file__}")
        
        # 测试prng
        if hasattr(spaces, 'prng'):
            print(f"  spaces.prng: 存在")
            print(f"  spaces.prng.np_random: {spaces.prng.np_random}")
        
        print("\n✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")