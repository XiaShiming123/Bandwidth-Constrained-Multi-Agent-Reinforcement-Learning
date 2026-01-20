#!/usr/bin/env python3
"""
简化版算法框架测试
"""

import numpy as np
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.utils.config import get_default_config


class SimpleTestEnv:
    """简单的测试环境"""
    
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.obs_dim = 10
        self.action_dim = 5
        self.state_dim = self.obs_dim * num_agents
        self.max_steps = 50
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
    
    def step(self, actions):
        self.current_step += 1
        rewards = [np.random.uniform(-1, 1) for _ in range(self.num_agents)]
        dones = [self.current_step >= self.max_steps] * self.num_agents
        next_obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
        return next_obs, rewards, dones, {}
    
    def get_agent_position(self, agent_id):
        return np.random.randn(2)


def test_imports():
    """测试导入"""
    print("Testing imports...")
    
    try:
        from experiments.algorithms.adaptive_comm import AdaptiveComm
        from experiments.algorithms.mappo import MAPPO
        from experiments.algorithms.iacn import IACN
        from experiments.algorithms.sparse_comm import SparseComm
        from experiments.algorithms.full_comm import FullComm
        print("  OK - All algorithms imported")
        return True
    except Exception as e:
        print(f"  FAIL - Import error: {e}")
        return False


def test_config():
    """测试配置"""
    print("\nTesting configuration...")
    
    try:
        config = get_default_config()
        config.device = "cpu"
        print(f"  OK - Config loaded")
        print(f"    Environment: {config.experiment.env}")
        print(f"    Training episodes: {config.training.num_episodes}")
        return True
    except Exception as e:
        print(f"  FAIL - Config error: {e}")
        return False


def test_algorithm_creation():
    """测试算法创建"""
    print("\nTesting algorithm creation...")
    
    config = get_default_config()
    config.device = "cpu"
    env = SimpleTestEnv(num_agents=2)
    
    algorithms = [
        ("AdaptiveComm", "experiments.algorithms.adaptive_comm", "AdaptiveComm"),
        ("MAPPO", "experiments.algorithms.mappo", "MAPPO"),
        ("IACN", "experiments.algorithms.iacn", "IACN"),
        ("SparseComm", "experiments.algorithms.sparse_comm", "SparseComm"),
        ("FullComm", "experiments.algorithms.full_comm", "FullComm")
    ]
    
    success_count = 0
    for name, module_name, class_name in algorithms:
        try:
            module = __import__(module_name, fromlist=[''])
            algorithm_class = getattr(module, class_name)
            algorithm = algorithm_class(env, config)
            print(f"  OK - {name} created")
            success_count += 1
        except Exception as e:
            print(f"  FAIL - {name}: {e}")
    
    return success_count == len(algorithms)


def test_simple_training():
    """测试简单训练"""
    print("\nTesting simple training...")
    
    try:
        from experiments.algorithms.adaptive_comm import AdaptiveComm
        
        config = get_default_config()
        config.device = "cpu"
        config.training.num_episodes = 2
        config.training.max_steps = 5
        
        env = SimpleTestEnv(num_agents=2)
        algorithm = AdaptiveComm(env, config)
        
        # 简单训练循环
        for episode in range(config.training.num_episodes):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(config.training.max_steps):
                actions, log_probs, values = algorithm.select_action(obs, training=True)
                next_obs, rewards, dones, _ = env.step(actions)
                
                # 存储经验
                algorithm.store_transition(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=dones,
                    values=values,
                    log_probs=log_probs
                )
                
                obs = next_obs
                episode_reward += np.mean(rewards)
            
            print(f"  Episode {episode + 1}: reward = {episode_reward:.2f}")
        
        print("  OK - Training completed")
        return True
        
    except Exception as e:
        print(f"  FAIL - Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Multi-Agent Communication Framework Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Algorithm Creation", test_algorithm_creation),
        ("Simple Training", test_simple_training)
    ]
    
    results = []
    for name, test in tests:
        print(f"\n[Test] {name}")
        print("-" * 40)
        try:
            passed = test()
            results.append((name, passed))
        except Exception as e:
            print(f"Test error: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed_count = 0
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20} {status}")
        if passed:
            passed_count += 1
    
    total = len(results)
    print(f"\nPassed: {passed_count}/{total}")
    
    if passed_count == total:
        print("\nAll tests passed! Framework is working.")
        return True
    else:
        print("\nSome tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)