#!/usr/bin/env python3
"""
测试远程服务器MPE环境修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟远程服务器上的MPE场景
class MockMPEScenarios:
    """模拟远程服务器上的MPE场景模块"""
    
    @staticmethod
    def load(scenario_file):
        """加载场景"""
        scenario_name = scenario_file.replace('.py', '')
        
        if scenario_name in ['imp', 'load', 'osp']:
            # 返回一个模拟的场景类
            class MockScenario:
                class Scenario:
                    def make_world(self):
                        class World:
                            def __init__(self):
                                self.dim_c = 0  # 注意：这里dim_c=0会导致问题
                                self.dim_p = 2
                                self.agents = []
                                self.landmarks = []
                                
                        world = World()
                        
                        # 添加3个智能体
                        class Agent:
                            def __init__(self):
                                self.name = ''
                                self.silent = True  # 所有智能体都是静默的
                                self.collide = True
                                self.size = 0.08
                                self.state = type('State', (), {'p_pos': [0, 0], 'p_vel': [0, 0], 'c': []})()
                                self.action = type('Action', (), {'u': [0, 0], 'c': []})()
                        
                        world.agents = [Agent() for i in range(3)]
                        for i, agent in enumerate(world.agents):
                            agent.name = f'agent {i}'
                        
                        # 添加地标
                        class Landmark:
                            def __init__(self):
                                self.name = ''
                                self.collide = False
                                self.movable = False
                                self.state = type('State', (), {'p_pos': [0, 0], 'p_vel': [0, 0]})()
                        
                        world.landmarks = [Landmark() for i in range(3)]
                        for i, landmark in enumerate(world.landmarks):
                            landmark.name = f'landmark {i}'
                        
                        return world
                    
                    def reset_world(self, world):
                        pass
                    
                    def reward(self, agent, world):
                        return 0.0
                    
                    def observation(self, agent, world):
                        return [0.0] * 10
            
            return MockScenario()
        else:
            raise FileNotFoundError(f"No such file or directory: '{scenario_file}'")
    
    @staticmethod
    def __dir__():
        return ['imp', 'load', 'osp']

# 模拟multiagent模块
class MockMultiAgent:
    class environment:
        class MultiAgentEnv:
            def __init__(self, world, reset_callback, reward_callback, observation_callback, 
                         info_callback=None, done_callback=None, shared_viewer=True):
                self.world = world
                self.n_agents = len(world.agents)
                self.observation_space = []
                self.action_space = []
                
                # 模拟创建动作空间（这里会触发AssertionError）
                import gym
                from gym import spaces
                
                for agent in world.agents:
                    # 物理动作空间
                    u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
                    
                    # 通信动作空间 - 这里会失败，因为world.dim_c=0
                    c_action_space = spaces.Discrete(world.dim_c)  # 这会触发AssertionError
                    
                    # 总动作空间
                    if not agent.silent:
                        act_space = spaces.Tuple([u_action_space, c_action_space])
                    else:
                        act_space = u_action_space
                    
                    self.action_space.append(act_space)
                    
                    # 观测空间
                    obs_dim = len(observation_callback(agent, world))
                    self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            
            def reset(self):
                return [[0.0] * 10 for _ in range(self.n_agents)]
            
            def step(self, actions):
                return [[0.0] * 10 for _ in range(self.n_agents)], [0.0] * self.n_agents, [False] * self.n_agents, {}
            
            def close(self):
                pass

# 测试修复
import numpy as np

def test_with_mock():
    """使用模拟的MPE环境测试修复"""
    print("测试模拟的MPE环境...")
    
    # 临时替换模块
    import sys
    original_modules = {}
    
    try:
        # 保存原始模块
        if 'multiagent' in sys.modules:
            original_modules['multiagent'] = sys.modules['multiagent']
        if 'multiagent.scenarios' in sys.modules:
            original_modules['multiagent.scenarios'] = sys.modules['multiagent.scenarios']
        if 'multiagent.environment' in sys.modules:
            original_modules['multiagent.environment'] = sys.modules['multiagent.environment']
        
        # 替换为模拟模块
        sys.modules['multiagent.scenarios'] = MockMPEScenarios
        sys.modules['multiagent.environment'] = MockMultiAgent.environment
        
        # 导入我们的环境封装
        from experiments.environments.multiagent_env import MultiAgentEnvWrapper
        
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
        
        config = MockConfig()
        
        print("1. 测试MPE_Navigation环境创建:")
        try:
            env = MultiAgentEnvWrapper("MPE_Navigation", config)
            print(f"   ✓ 环境创建成功")
            print(f"     智能体数量: {env.num_agents}")
            
            # 测试重置
            obs = env.reset()
            print(f"   ✓ 重置成功")
            
            # 测试一步
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
            obs, rewards, dones, info = env.step(actions)
            print(f"   ✓ 步进成功")
            
            env.close()
            print(f"   ✓ 环境关闭成功")
            
            return True
            
        except Exception as e:
            print(f"   ✗ 环境创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # 恢复原始模块
        for name, module in original_modules.items():
            sys.modules[name] = module
        
        # 删除模拟模块
        if 'multiagent.scenarios' in sys.modules and 'multiagent.scenarios' not in original_modules:
            del sys.modules['multiagent.scenarios']
        if 'multiagent.environment' in sys.modules and 'multiagent.environment' not in original_modules:
            del sys.modules['multiagent.environment']

if __name__ == "__main__":
    print("=" * 60)
    print("测试远程服务器MPE环境修复")
    print("=" * 60)
    
    if test_with_mock():
        print("\n" + "=" * 60)
        print("✓ 测试通过！修复应该能在远程服务器上工作。")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ 测试失败。")
        print("=" * 60)
        sys.exit(1)