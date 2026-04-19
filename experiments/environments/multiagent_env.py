#!/usr/bin/env python3
"""
多智能体环境封装，支持MPE和SMAC环境
"""

import numpy as np
import gym
from gym import spaces
import torch
from typing import Dict, List, Tuple, Any, Optional


def normalize_observation(obs, obs_space):
    """标准化观测值"""
    if isinstance(obs_space, spaces.Box):
        # 线性归一化到[0, 1]
        low = obs_space.low
        high = obs_space.high
        obs_normalized = (obs - low) / (high - low + 1e-8)
        return obs_normalized
    return obs


class MultiAgentEnvWrapper:
    """多智能体环境封装类"""
    
    def __init__(self, env_name: str, config):
        """初始化环境"""
        self.env_name = env_name
        self.config = config
        
        # 创建环境
        if env_name.startswith("MPE"):
            self._init_mpe_env(env_name)
        elif env_name.startswith("SMAC"):
            self._init_smac_env(env_name)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        # 环境参数
        self.num_agents = self.env.n if hasattr(self.env, 'n') else (self.env.n_agents if hasattr(self.env, 'n_agents') else self.env.num_agents)
        self.max_steps = config.environment.max_steps
        self.current_step = 0
        
        # 通信约束
        self.comm_constraints = {
            'bandwidth_limit': config.communication.bandwidth_limit,  # KB/step
            'latency': config.communication.latency,  # steps
            'packet_loss': config.communication.packet_loss,  # probability
        }
        
        # 获取观测和动作空间
        self._setup_spaces()
        
        # 通信历史
        self.comm_history = []
        
        print(f"Environment: {env_name}")
        print(f"Number of agents: {self.num_agents}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"State dim: {self.state_dim}")
    
    def _init_mpe_env(self, env_name: str):
        """初始化MPE环境"""
        try:
            from multiagent.environment import MultiAgentEnv
            import multiagent.scenarios as scenarios
        except ImportError:
            raise ImportError("Please install multiagent-particle-envs: pip install -e multiagent-particle-envs")
        
        # 根据环境名称选择场景
        if env_name == "MPE_Navigation":
            scenario_name = "simple_navigation"
        elif env_name == "MPE_PredatorPrey":
            scenario_name = "simple_predator_prey"
        else:
            scenario_name = "simple"
        
        # 加载场景
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        
        # 创建环境
        self.env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            info_callback=scenario.info_callback if hasattr(scenario, 'info_callback') else None,
            done_callback=scenario.done_callback if hasattr(scenario, 'done_callback') else None,
            shared_viewer=False
        )
    
    def _init_smac_env(self, env_name: str):
        """初始化SMAC环境"""
        try:
            from smac.env import StarCraft2Env
        except ImportError:
            raise ImportError("Please install SMAC: pip install git+https://github.com/oxwhirl/smac.git")
        
        # 提取地图名称
        map_name = env_name.split("_")[1]
        
        # 创建环境
        self.env = StarCraft2Env(
            map_name=map_name,
            step_mul=self.config.environment.step_mul,
            difficulty=self.config.environment.difficulty,
            replay_dir=self.config.environment.replay_dir
        )
    
    def _setup_spaces(self):
        """设置观测和动作空间"""
        if hasattr(self.env, 'observation_space'):
            # MPE环境
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            
            # 处理观测空间
            if isinstance(self.observation_space, list):
                self.obs_dim = self.observation_space[0].shape[0]
            else:
                self.obs_dim = self.observation_space.shape[0]
                
            # 处理动作空间
            if isinstance(self.action_space, list):
                # 获取第一个智能体的动作空间
                agent_action_space = self.action_space[0]
                
                # 检查是否是MultiDiscrete类型
                if hasattr(agent_action_space, 'high') and hasattr(agent_action_space, 'low'):
                    # MultiDiscrete动作空间
                    # 计算总动作数量：sum(high - low + 1)
                    self.action_dim = int(np.sum(agent_action_space.high - agent_action_space.low + 1))
                elif hasattr(agent_action_space, 'n'):
                    # Discrete动作空间
                    self.action_dim = agent_action_space.n
                elif hasattr(agent_action_space, 'shape'):
                    # Box动作空间
                    self.action_dim = agent_action_space.shape[0]
                else:
                    # 默认处理
                    self.action_dim = len(self.action_space)
            else:
                # 动作空间不是列表
                if hasattr(self.action_space, 'high') and hasattr(self.action_space, 'low'):
                    self.action_dim = int(np.sum(self.action_space.high - self.action_space.low + 1))
                elif hasattr(self.action_space, 'n'):
                    self.action_dim = self.action_space.n
                elif hasattr(self.action_space, 'shape'):
                    self.action_dim = self.action_space.shape[0]
                else:
                    self.action_dim = 1
                
            # 状态空间（全局观测）
            self.state_dim = self.obs_dim * self.num_agents
            
        elif hasattr(self.env, 'get_obs_size'):
            # SMAC环境
            self.obs_dim = self.env.get_obs_size()
            self.state_dim = self.env.get_state_size()
            self.action_dim = self.env.get_total_actions()
            
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
            self.action_space = spaces.Discrete(self.action_dim)
        else:
            raise ValueError("Unsupported environment type")
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.comm_history = []
        
        if hasattr(self.env, 'reset'):
            obs = self.env.reset()
        else:
            obs = self.env.reset()[0]
        
        # 标准化观测
        if isinstance(obs, list):
            obs = [normalize_observation(o, self.observation_space) for o in obs]
        else:
            obs = normalize_observation(obs, self.observation_space)
        
        return obs
    
    def step(self, actions):
        """执行一步动作"""
        self.current_step += 1
        
        # 执行动作
        if hasattr(self.env, 'step'):
            obs, rewards, dones, info = self.env.step(actions)
        else:
            obs, rewards, dones, info = self.env.step(actions)[:4]
        
        # 标准化观测
        if isinstance(obs, list):
            obs = [normalize_observation(o, self.observation_space) for o in obs]
        else:
            obs = normalize_observation(obs, self.observation_space)
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            dones = [True] * self.num_agents if isinstance(dones, list) else True
        
        # 记录通信（如果有）
        if hasattr(self, 'last_communication'):
            self.comm_history.append({
                'step': self.current_step,
                'communication': self.last_communication
            })
        
        return obs, rewards, dones, info
    
    def get_state(self):
        """获取全局状态"""
        if hasattr(self.env, 'get_state'):
            return self.env.get_state()
        else:
            # 对于MPE环境，获取所有智能体的观测并拼接
            if hasattr(self.env, 'world') and hasattr(self.env.world, 'agents'):
                # 获取所有智能体的观测
                obs_list = []
                for agent in self.env.world.agents:
                    obs = self.env._get_obs(agent)
                    obs_list.append(obs)
                return np.concatenate(obs_list)
            else:
                # 备用方法：尝试直接获取观测
                try:
                    obs = self.env._get_obs()
                    if isinstance(obs, list):
                        return np.concatenate(obs)
                    else:
                        return obs
                except:
                    # 如果失败，返回零向量
                    return np.zeros(self.state_dim)
    
    def get_agent_position(self, agent_id: int) -> np.ndarray:
        """获取智能体位置（用于构建通信拓扑）"""
        if hasattr(self.env, 'world'):
            # MPE环境
            return self.env.world.agents[agent_id].state.p_pos
        elif hasattr(self.env, 'get_unit_by_id'):
            # SMAC环境
            unit = self.env.get_unit_by_id(agent_id)
            return np.array([unit.pos.x, unit.pos.y])
        else:
            # 默认返回随机位置
            return np.random.randn(2)
    
    def get_agent_distance(self, agent_id1: int, agent_id2: int) -> float:
        """计算两个智能体之间的距离"""
        pos1 = self.get_agent_position(agent_id1)
        pos2 = self.get_agent_position(agent_id2)
        return np.linalg.norm(pos1 - pos2)
    
    def apply_communication_constraints(self, messages: List[Dict]) -> List[Dict]:
        """应用通信约束"""
        constrained_messages = []
        
        for msg in messages:
            # 模拟丢包
            if np.random.random() < self.comm_constraints['packet_loss']:
                continue
            
            # 模拟带宽限制（简化：限制消息大小）
            if 'size' in msg:
                max_size = self.comm_constraints['bandwidth_limit'] * 1024  # 转换为字节
                if msg['size'] > max_size:
                    # 压缩消息（简化：随机丢弃部分内容）
                    msg['content'] = msg['content'][:max_size//4]  # 假设每个元素4字节
                    msg['size'] = len(msg['content']) * 4
            
            # 模拟延迟（简化：标记延迟）
            msg['delayed'] = np.random.random() < 0.1  # 10%的概率延迟
            
            constrained_messages.append(msg)
        
        return constrained_messages
    
    def render(self, mode='human'):
        """渲染环境"""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None
    
    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @property
    def episode_length(self):
        """获取当前回合步数"""
        return self.current_step
    
    def get_communication_stats(self) -> Dict:
        """获取通信统计信息"""
        if not self.comm_history:
            return {
                'total_messages': 0,
                'total_bytes': 0,
                'avg_messages_per_step': 0,
                'avg_bytes_per_step': 0
            }
        
        total_messages = 0
        total_bytes = 0
        
        for comm in self.comm_history:
            if 'communication' in comm:
                comm_data = comm['communication']
                total_messages += comm_data.get('num_messages', 0)
                total_bytes += comm_data.get('total_bytes', 0)
        
        return {
            'total_messages': total_messages,
            'total_bytes': total_bytes,
            'avg_messages_per_step': total_messages / len(self.comm_history),
            'avg_bytes_per_step': total_bytes / len(self.comm_history)
        }