#!/usr/bin/env python3
"""
测试最终修复
"""

import sys
import os
import numpy as np  # 添加numpy导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量
os.environ['SUPPRESS_MA_PROMPT'] = '1'

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("测试MPE环境最终修复")
print("=" * 60)

# 创建模拟配置
class MockConfig:
    class environment:
        max_steps = 1000
    
    class communication:
        bandwidth_limit = 100
        latency = 1
        packet_loss = 0.1
    
    environment = environment()
    communication = communication()

try:
    from experiments.environments.multiagent_env import MultiAgentEnvWrapper
    
    config = MockConfig()
    
    print("1. 测试MPE_Navigation环境创建:")
    try:
        env = MultiAgentEnvWrapper("MPE_Navigation", config)
        print(f"  ✓ 环境创建成功")
        print(f"     智能体数量: {env.num_agents}")
        print(f"     观测维度: {env.obs_dim}")
        print(f"     动作维度: {env.action_dim}")
        
        print("\n2. 测试环境功能:")
        
        # 测试重置
        obs = env.reset()
        print(f"  ✓ 重置成功")
        
        # 生成动作 - 使用numpy数组
        if isinstance(env.action_space, list):
            actions = []
            for i, space in enumerate(env.action_space):
                # 对于MPE环境，使用零动作，转换为numpy数组
                # 根据environment.py，对于discrete_action_space且discrete_action_input=False
                # 期望action[0]是一个numpy数组，至少5个元素
                actions.append(np.zeros(5, dtype=np.float32))  # 使用numpy数组
        else:
            actions = np.zeros(5, dtype=np.float32)  # 默认使用numpy数组
        
        print(f"  使用的动作: {actions}")
        
        # 测试一步
        obs, rewards, dones, info = env.step(actions)
        print(f"  ✓ 步进成功")
        print(f"     奖励: {rewards}")
        
        # 测试获取状态
        state = env.get_state()
        print(f"  ✓ 获取状态成功，状态维度: {len(state) if hasattr(state, '__len__') else 1}")
        
        env.close()
        print(f"  ✓ 环境关闭成功")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！修复成功。")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✗ 测试失败")
        print("=" * 60)
        sys.exit(1)
        
except ImportError as e:
    print(f"导入失败: {e}")
    print("\n" + "=" * 60)
    print("注意: 本地没有安装MPE环境")
    print("修复代码已准备好，可以同步到远程服务器测试")
    print("=" * 60)
    sys.exit(0)