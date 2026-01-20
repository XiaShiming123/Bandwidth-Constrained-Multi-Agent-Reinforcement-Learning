#!/usr/bin/env python3
"""
测试MPE修复
"""

import os
import sys
import warnings

# 设置
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_MA_PROMPT'] = '1'

print("=" * 60)
print("测试MPE修复")
print("=" * 60)
print()

# 修复Gym兼容性
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
    print(f"✅ 使用Gymnasium")
    
    import numpy as np
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
except:
    import gym
    print(f"✅ 使用Gym")

print()

# 测试MPE导入
try:
    import multiagent.scenarios as scenarios
    
    # 列出所有场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"MPE可用场景 ({len(scenario_names)}个):")
    for i, name in enumerate(scenario_names):
        print(f"  {i+1:2d}. {name}")
    
    # 测试加载simple场景（应该存在）
    print(f"\n测试加载'simple'场景:")
    try:
        scenario = scenarios.load("simple.py").Scenario()
        world = scenario.make_world()
        print(f"  ✅ 'simple'场景加载成功")
        print(f"     智能体数量: {len(world.agents)}")
        print(f"     地标数量: {len(world.landmarks)}")
    except Exception as e:
        print(f"  ❌ 'simple'场景加载失败: {e}")
    
    # 测试加载其他可能场景
    test_scenes = ["simple_spread", "simple_tag", "simple_adversary", "simple_reference"]
    print(f"\n测试其他场景:")
    for scene in test_scenes:
        if scene in scenario_names:
            try:
                scenario = scenarios.load(scene + ".py").Scenario()
                world = scenario.make_world()
                print(f"  ✅ {scene}: 成功 (agents: {len(world.agents)})")
            except Exception as e:
                print(f"  ❌ {scene}: 失败 - {str(e)[:50]}")
        else:
            print(f"  ⚠️  {scene}: 不存在")
    
except ImportError as e:
    print(f"❌ MPE导入失败: {e}")
    sys.exit(1)

print()

# 测试环境封装
print("测试环境封装:")
try:
    sys.path.append('.')
    from experiments.environments.multiagent_env import MultiAgentEnvWrapper
    
    # 创建简单配置
    class SimpleConfig:
        class experiment:
            env = "MPE_Navigation"
            output_dir = "test"
            device = "cpu"
        class environment:
            max_steps = 100
            num_agents = 3
        class communication:
            bandwidth_limit = 10.0
            latency = 1
            packet_loss = 0.1
        class training:
            num_episodes = 100
            max_steps = 100
            buffer_size = 1024
            batch_size = 32
            gamma = 0.99
            gae_lambda = 0.95
            ppo_clip = 0.2
            value_coef = 0.5
            entropy_coef = 0.01
            max_grad_norm = 0.5
            learning_rate = 3e-4
            log_interval = 10
        class evaluation:
            num_episodes = 5
            success_threshold = 50.0
        class adaptive_comm:
            comm_dim = 32
            comm_cost_coef = 0.1
            sparsity_threshold = 0.3
            topology_update_freq = 10
            base_neighbors = 2
    
    config = SimpleConfig()
    
    # 测试MPE_Navigation
    print(f"\n创建MPE_Navigation环境:")
    env = MultiAgentEnvWrapper("MPE_Navigation", config)
    print(f"  ✅ 环境创建成功")
    print(f"     智能体数量: {env.num_agents}")
    print(f"     观测维度: {env.obs_dim}")
    print(f"     动作维度: {env.action_dim}")
    
    # 测试重置
    obs = env.reset()
    print(f"  ✅ 重置成功")
    
    # 测试一步
    if isinstance(obs, list):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
    else:
        actions = env.action_space.sample()
    
    next_obs, rewards, dones, info = env.step(actions)
    print(f"  ✅ 步进成功")
    print(f"     奖励: {rewards}")
    
    env.close()
    print(f"  ✅ 环境测试完成")
    
except Exception as e:
    print(f"❌ 环境封装测试失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 检查GPU
print("检查GPU:")
try:
    import torch
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {name} ({mem:.1f}GB)")
except Exception as e:
    print(f"  ❌ 检查GPU失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)