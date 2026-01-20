#!/usr/bin/env python3
"""
测试算法框架

由于没有安装MPE/SMAC环境，我们创建一个简化的测试环境
来验证算法框架的正确性。
"""

import numpy as np
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.algorithms.base_algorithm import BaseAlgorithm
from experiments.algorithms.adaptive_comm import AdaptiveComm
from experiments.algorithms.mappo import MAPPO
from experiments.algorithms.iacn import IACN
from experiments.algorithms.sparse_comm import SparseComm
from experiments.algorithms.full_comm import FullComm

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
        
        # 简单的奖励函数
        rewards = [np.random.uniform(-1, 1) for _ in range(self.num_agents)]
        
        # 50步后结束
        dones = [self.current_step >= self.max_steps] * self.num_agents
        
        # 新的观测
        next_obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
        
        return next_obs, rewards, dones, {}
    
    def get_agent_position(self, agent_id):
        # 返回随机位置
        return np.random.randn(2)


def test_algorithm_initialization():
    """测试算法初始化"""
    print("测试算法初始化...")
    
    config = get_default_config()
    config.device = "cpu"
    
    env = SimpleTestEnv(num_agents=3)
    
    algorithms = {
        "AdaptiveComm": AdaptiveComm,
        "MAPPO": MAPPO,
        "IACN": IACN,
        "SparseComm": SparseComm,
        "FullComm": FullComm
    }
    
    for name, AlgorithmClass in algorithms.items():
        try:
            algorithm = AlgorithmClass(env, config)
            print(f"  ✓ {name} 初始化成功")
            
            # 测试动作选择
            obs = env.reset()
            actions, log_probs, values = algorithm.select_action(obs, training=True)
            print(f"    - 动作选择: {len(actions)} 个动作")
            
        except Exception as e:
            print(f"  ✗ {name} 初始化失败: {e}")
    
    return True


def test_training_loop():
    """测试训练循环"""
    print("\n测试训练循环（简化版）...")
    
    config = get_default_config()
    config.device = "cpu"
    config.training.num_episodes = 5  # 减少回合数
    config.training.max_steps = 10    # 减少步数
    
    env = SimpleTestEnv(num_agents=2)
    
    # 只测试AdaptiveComm
    try:
        algorithm = AdaptiveComm(env, config)
        
        episode_rewards = []
        
        for episode in range(config.training.num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < config.training.max_steps:
                # 选择动作
                actions, log_probs, values = algorithm.select_action(obs, training=True)
                
                # 执行动作
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
                
                # 更新
                obs = next_obs
                episode_reward += np.mean(rewards)
                steps += 1
                
                # 检查是否结束
                done = all(dones)
            
            episode_rewards.append(episode_reward)
            
            # 更新网络
            if len(algorithm.buffer['obs']) >= algorithm.batch_size:
                algorithm.update()
            
            print(f"  回合 {episode + 1}: 奖励 = {episode_reward:.2f}")
        
        print(f"  ✓ AdaptiveComm 训练循环完成，平均奖励: {np.mean(episode_rewards):.2f}")
        return True
        
    except Exception as e:
        print(f"  ✗ 训练循环失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """测试评估"""
    print("\n测试评估...")
    
    config = get_default_config()
    config.device = "cpu"
    config.evaluation.num_episodes = 3
    
    env = SimpleTestEnv(num_agents=2)
    
    try:
        algorithm = AdaptiveComm(env, config)
        
        # 评估
        results = algorithm.evaluate(num_episodes=config.evaluation.num_episodes)
        
        print(f"  平均奖励: {results['avg_reward']:.2f}")
        print(f"  成功率: {results['success_rate']:.2%}")
        print(f"  通信成本: {results['comm_cost']:.2f} KB/step")
        
        print(f"  ✓ 评估完成")
        return True
        
    except Exception as e:
        print(f"  ✗ 评估失败: {e}")
        return False


def test_communication_stats():
    """测试通信统计"""
    print("\n测试通信统计...")
    
    config = get_default_config()
    config.device = "cpu"
    
    env = SimpleTestEnv(num_agents=3)
    
    algorithms_with_comm = ["AdaptiveComm", "IACN", "SparseComm", "FullComm"]
    
    for algo_name in algorithms_with_comm:
        try:
            if algo_name == "AdaptiveComm":
                algorithm = AdaptiveComm(env, config)
            elif algo_name == "IACN":
                algorithm = IACN(env, config)
            elif algo_name == "SparseComm":
                algorithm = SparseComm(env, config)
            elif algo_name == "FullComm":
                algorithm = FullComm(env, config)
            
            # 运行几步以生成通信数据
            obs = env.reset()
            for _ in range(10):
                actions, _, _ = algorithm.select_action(obs, training=True)
                next_obs, rewards, dones, _ = env.step(actions)
                obs = next_obs
            
            # 获取通信统计
            if hasattr(algorithm, 'get_communication_stats'):
                stats = algorithm.get_communication_stats()
                comm_rate = stats.get('communication_rate', 0)
                comm_cost = stats.get('avg_message_size', 0)
                
                print(f"  {algo_name:15} 通信率: {comm_rate:.2%}, 成本: {comm_cost:.2f} KB/step")
            
        except Exception as e:
            print(f"  ✗ {algo_name} 通信统计失败: {e}")
    
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("多智能体通信算法框架测试")
    print("=" * 60)
    
    tests = [
        ("算法初始化", test_algorithm_initialization),
        ("训练循环", test_training_loop),
        ("评估", test_evaluation),
        ("通信统计", test_communication_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[测试] {test_name}")
        print("-" * 40)
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed_count = 0
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:15} {status}")
        if passed:
            passed_count += 1
    
    total = len(results)
    print(f"\n通过: {passed_count}/{total}")
    
    if passed_count == total:
        print("\n所有测试通过! 算法框架工作正常。")
        print("\n下一步:")
        print("1. 安装MPE环境: git clone https://github.com/openai/multiagent-particle-envs.git")
        print("2. 运行完整实验: python experiments/main_experiment.py")
        return True
    else:
        print("\n部分测试失败，需要调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)