#!/usr/bin/env python3
"""
简单测试MPE环境修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 创建模拟配置
class MockConfig:
    class environment:
        max_steps = 1000
        step_mul = 8
        difficulty = "7"
        replay_dir = "./replays"
    
    class communication:
        bandwidth_limit = 100  # KB/step
        latency = 1  # steps
        packet_loss = 0.1  # probability
    
    environment = environment()
    communication = communication()

def test_scenario_loading():
    """测试场景加载"""
    print("测试场景加载...")
    try:
        import multiagent.scenarios as scenarios
        
        # 测试simple_navigation场景
        print("1. 测试simple_navigation场景:")
        scenario = scenarios.load("simple_navigation.py").Scenario()
        world = scenario.make_world()
        print(f"   成功创建世界")
        print(f"   智能体数量: {len(world.agents)}")
        print(f"   通信维度: {world.dim_c}")
        print(f"   智能体是否静默: {[agent.silent for agent in world.agents]}")
        
        # 测试simple场景
        print("\n2. 测试simple场景:")
        scenario = scenarios.load("simple.py").Scenario()
        world = scenario.make_world()
        print(f"   成功创建世界")
        print(f"   智能体数量: {len(world.agents)}")
        print(f"   通信维度: {world.dim_c}")
        
        # 测试simple_tag场景
        print("\n3. 测试simple_tag场景:")
        scenario = scenarios.load("simple_tag.py").Scenario()
        world = scenario.make_world()
        print(f"   成功创建世界")
        print(f"   智能体数量: {len(world.agents)}")
        print(f"   通信维度: {world.dim_c}")
        
        return True
    except Exception as e:
        print(f"场景加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_wrapper():
    """测试环境封装"""
    print("\n测试环境封装...")
    try:
        from experiments.environments.multiagent_env import MultiAgentEnvWrapper
        
        config = MockConfig()
        
        # 测试MPE_Navigation
        print("1. 测试MPE_Navigation:")
        env = MultiAgentEnvWrapper("MPE_Navigation", config)
        print(f"   环境创建成功")
        print(f"   智能体数量: {env.num_agents}")
        print(f"   观测维度: {env.obs_dim}")
        print(f"   动作维度: {env.action_dim}")
        
        # 测试重置
        obs = env.reset()
        print(f"   重置成功，观测类型: {type(obs)}")
        
        # 测试一步
        if isinstance(env.action_space, list):
            actions = [space.sample() for space in env.action_space]
        else:
            actions = env.action_space.sample()
        
        obs, rewards, dones, info = env.step(actions)
        print(f"   步进成功，奖励: {rewards}")
        
        env.close()
        
        # 测试MPE_PredatorPrey (simple_tag)
        print("\n2. 测试MPE_PredatorPrey:")
        env = MultiAgentEnvWrapper("MPE_PredatorPrey", config)
        print(f"   环境创建成功")
        print(f"   智能体数量: {env.num_agents}")
        env.close()
        
        return True
    except Exception as e:
        print(f"环境封装测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试MPE环境修复")
    print("=" * 60)
    
    success = True
    
    if not test_scenario_loading():
        success = False
    
    if not test_environment_wrapper():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("测试通过！MPE环境修复成功。")
        sys.exit(0)
    else:
        print("测试失败。")
        sys.exit(1)