#!/usr/bin/env python3
"""
测试MPE场景可用性
"""

import os
import sys
import warnings

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_MA_PROMPT'] = '1'

print("=" * 60)
print("测试MPE场景")
print("=" * 60)

# 修复Gym兼容性
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
    import numpy as np
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
except:
    import gym

# 导入MPE
try:
    import multiagent.scenarios as scenarios
    
    # 列出所有场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"\n可用场景 ({len(scenario_names)}个):")
    for i, name in enumerate(scenario_names):
        print(f"  {i+1:2d}. {name}")
    
    # 测试加载每个场景
    print("\n测试场景加载:")
    for scenario_name in scenario_names[:10]:  # 只测试前10个
        try:
            scenario = scenarios.load(scenario_name + ".py").Scenario()
            world = scenario.make_world()
            print(f"  ✅ {scenario_name}: 成功 (agents: {len(world.agents)})")
        except Exception as e:
            print(f"  ❌ {scenario_name}: 失败 - {str(e)[:50]}")
    
    # 测试我们的环境封装
    print("\n测试环境封装:")
    from experiments.environments.multiagent_env import MultiAgentEnvWrapper
    
    # 创建简单配置
    class SimpleConfig:
        class experiment:
            env = "MPE_Navigation"
            output_dir = "test"
            device = "cpu"
        class environment:
            max_steps = 100
            step_mul = 8
            difficulty = "7"
            replay_dir = "replays"
        class communication:
            bandwidth_limit = 10.0
            latency = 1
            packet_loss = 0.1
        class training:
            num_episodes = 1000
            max_steps = 100
            buffer_size = 2048
            batch_size = 64
            gamma = 0.99
            gae_lambda = 0.95
            ppo_clip = 0.2
            value_coef = 0.5
            entropy_coef = 0.01
            max_grad_norm = 0.5
            learning_rate = 3e-4
            log_interval = 10
        class evaluation:
            num_episodes = 10
            success_threshold = 50.0
    
    config = SimpleConfig()
    
    # 测试MPE_Navigation
    try:
        env = MultiAgentEnvWrapper("MPE_Navigation", config)
        print(f"  ✅ MPE_Navigation环境创建成功")
        print(f"     智能体数量: {env.num_agents}")
        print(f"     观测维度: {env.obs_dim}")
        print(f"     动作维度: {env.action_dim}")
        
        # 测试重置
        obs = env.reset()
        print(f"     重置成功，观测类型: {type(obs)}")
        
        # 测试一步
        if isinstance(obs, list):
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
        else:
            actions = env.action_space.sample()
        
        next_obs, rewards, dones, info = env.step(actions)
        print(f"     步进成功，奖励: {rewards}")
        
        env.close()
        print("  ✅ 环境测试完成")
        
    except Exception as e:
        print(f"  ❌ MPE_Navigation环境创建失败: {e}")
        import traceback
        traceback.print_exc()
    
except ImportError as e:
    print(f"❌ MPE导入失败: {e}")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)