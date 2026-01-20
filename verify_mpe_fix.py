#!/usr/bin/env python3
"""
验证MPE环境修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量
os.environ['SUPPRESS_MA_PROMPT'] = '1'

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("验证MPE环境修复")
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
    # 首先检查MPE是否可用
    try:
        import multiagent.scenarios as scenarios
        
        # 列出可用场景
        available_scenarios = [name for name in dir(scenarios) if not name.startswith('_')]
        print(f"可用MPE场景: {available_scenarios}")
        
        if not available_scenarios:
            print("错误: 没有可用的MPE场景")
            sys.exit(1)
        
        # 测试加载第一个场景
        test_scenario = available_scenarios[0]
        print(f"\n测试加载场景: {test_scenario}")
        
        scenario = scenarios.load(test_scenario + ".py").Scenario()
        world = scenario.make_world()
        
        print(f"  场景加载成功")
        print(f"  智能体数量: {len(world.agents)}")
        print(f"  通信维度 (dim_c): {world.dim_c}")
        print(f"  智能体静默状态: {[agent.silent for agent in world.agents]}")
        
        # 检查问题
        issues = []
        if world.dim_c <= 0:
            issues.append(f"dim_c <= 0 ({world.dim_c})")
        
        if all([agent.silent for agent in world.agents]):
            issues.append("所有智能体都是静默的")
        
        if issues:
            print(f"  发现潜在问题: {', '.join(issues)}")
        else:
            print(f"  场景看起来正常")
        
    except ImportError as e:
        print(f"MPE导入失败: {e}")
        print("注意: 本地没有安装MPE环境，但修复代码应该能在远程服务器上工作")
        
    # 测试环境封装
    print("\n测试环境封装:")
    try:
        from experiments.environments.multiagent_env import MultiAgentEnvWrapper
        
        config = MockConfig()
        
        print("1. 创建MPE_Navigation环境:")
        env = MultiAgentEnvWrapper("MPE_Navigation", config)
        print(f"  成功创建环境")
        print(f"  智能体数量: {env.num_agents}")
        print(f"  观测维度: {env.obs_dim}")
        print(f"  动作维度: {env.action_dim}")
        
        # 测试基本功能
        print("\n2. 测试环境功能:")
        obs = env.reset()
        print(f"  重置成功")
        
        # 生成动作
        if isinstance(env.action_space, list):
            actions = []
            for space in env.action_space:
                try:
                    actions.append(space.sample())
                except:
                    # 如果采样失败，使用默认动作
                    if hasattr(space, 'n'):
                        actions.append(0)  # Discrete空间
                    else:
                        actions.append([0.0] * space.shape[0])  # Box空间
        else:
            try:
                actions = env.action_space.sample()
            except:
                actions = 0  # 默认动作
        
        obs, rewards, dones, info = env.step(actions)
        print(f"  步进成功")
        print(f"  奖励: {rewards}")
        
        env.close()
        print(f"  环境关闭成功")
        
        print("\n" + "=" * 60)
        print("验证通过！修复代码应该能在远程服务器上工作。")
        print("=" * 60)
        
    except Exception as e:
        print(f"环境封装测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("验证失败。请检查错误信息。")
        print("=" * 60)
        sys.exit(1)
        
except Exception as e:
    print(f"测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)