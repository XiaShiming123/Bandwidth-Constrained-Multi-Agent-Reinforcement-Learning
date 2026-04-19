#!/usr/bin/env python3
"""
IACN (Independent Actor with Communication Network) 算法实现
基于原文[7]的实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from experiments.algorithms.base_algorithm import BaseAlgorithm


class IACN(BaseAlgorithm):
    """IACN算法：自适应通信频率，固定通信拓扑"""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # IACN特定参数
        self.comm_freq_base = config.iacn.comm_freq_base
        self.comm_freq_var = config.iacn.comm_freq_var
        self.comm_topology_type = config.iacn.topology_type  # 'full', 'nearest', 'random'
        
        # 通信网络
        self.comm_networks = nn.ModuleList([
            CommunicationNetwork(self.obs_dim, self.obs_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)
        
        # 通信历史
        self.comm_history = []
        self.last_comm_step = 0
        
        # 构建固定通信拓扑
        self.comm_topology = self._build_fixed_topology()
        
        print(f"IACN initialized with {self.comm_topology_type} topology")
        print(f"Communication frequency base: {self.comm_freq_base}")
    
    def _build_fixed_topology(self):
        """构建固定通信拓扑"""
        topology = {}
        
        if self.comm_topology_type == 'full':
            # 全连接拓扑
            for i in range(self.num_agents):
                topology[i] = [j for j in range(self.num_agents) if j != i]
                
        elif self.comm_topology_type == 'nearest':
            # 最近邻拓扑（基于初始位置）
            # 获取初始位置
            positions = []
            for i in range(self.num_agents):
                pos = self.env.get_agent_position(i)
                positions.append(pos)
            
            positions = np.array(positions)
            
            # 计算距离矩阵
            dist_matrix = np.zeros((self.num_agents, self.num_agents))
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])
            
            # 选择最近的k个邻居
            k = min(3, self.num_agents - 1)  # 最多3个邻居
            for i in range(self.num_agents):
                distances = [(j, dist_matrix[i, j]) for j in range(self.num_agents) if j != i]
                distances.sort(key=lambda x: x[1])
                topology[i] = [j for j, _ in distances[:k]]
                
        elif self.comm_topology_type == 'random':
            # 随机拓扑
            for i in range(self.num_agents):
                k = np.random.randint(1, min(4, self.num_agents))
                possible_neighbors = [j for j in range(self.num_agents) if j != i]
                topology[i] = list(np.random.choice(possible_neighbors, k, replace=False))
                
        else:
            raise ValueError(f"Unknown topology type: {self.comm_topology_type}")
        
        return topology
    
    def select_action(self, obs, training=True):
        """选择动作，包括自适应通信"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # 决定是否通信（基于自适应频率）
        should_communicate = self._should_communicate()
        
        # 如果通信，交换信息
        if should_communicate:
            updated_obs = self._communicate(obs_tensor)
            self.last_comm_step = self.total_steps
            
            # 记录通信
            self.comm_history.append({
                'step': self.total_steps,
                'should_communicate': True,
                'topology': self.comm_topology
            })
        else:
            updated_obs = obs_tensor
            
            # 记录非通信
            self.comm_history.append({
                'step': self.total_steps,
                'should_communicate': False
            })
        
        # 基于更新后的观测选择动作
        actions = []
        action_log_probs = []
        values = []
        
        for i in range(self.num_agents):
            agent_obs = updated_obs[i]
            
            # 通过行动策略网络
            action_dist = self.action_policies[i](agent_obs)
            
            if training:
                action = action_dist.sample()
            else:
                action = action_dist.mean
                
            action_log_prob = action_dist.log_prob(action)
            value = self.value_networks[i](agent_obs)
            
            # 在评估模式下使用 detach() 方法来移除梯度信息
            if training:
                actions.append(action.cpu().numpy())
            else:
                actions.append(action.detach().cpu().numpy())
            # 对log_prob进行求和，使其变为标量值
            action_log_probs.append(action_log_prob.sum())
            values.append(value)
        
        return actions, action_log_probs, values
    
    def _should_communicate(self):
        """决定是否进行通信"""
        # 基础通信频率
        base_prob = self.comm_freq_base
        
        # 基于价值函数方差调整频率
        if hasattr(self, 'last_values') and self.last_values:
            value_var = np.var(self.last_values)
            # 价值函数方差越大，通信频率越高
            adjusted_prob = base_prob * (1.0 + self.comm_freq_var * value_var)
            adjusted_prob = min(adjusted_prob, 0.9)  # 上限
        else:
            adjusted_prob = base_prob
        
        # 随机决定
        return np.random.random() < adjusted_prob
    
    def _communicate(self, obs_tensor):
        """执行通信"""
        updated_obs = obs_tensor.clone()
        
        # 每个智能体向邻居发送消息
        for i in range(self.num_agents):
            if i in self.comm_topology:
                neighbors = self.comm_topology[i]
                if neighbors:
                    # 编码观测
                    encoded_message = self.comm_networks[i].encode(obs_tensor[i])
                    
                    # 向每个邻居发送消息
                    for neighbor in neighbors:
                        # 解码消息
                        decoded_message = self.comm_networks[neighbor].decode(encoded_message)
                        
                        # 整合到邻居的观测中
                        updated_obs[neighbor] = updated_obs[neighbor] + 0.05 * decoded_message
        
        return updated_obs
    
    def train(self):
        """训练算法"""
        print(f"Training IACN for {self.config.training.num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        comm_rates = []
        
        for episode in range(self.config.training.num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            episode_comm_count = 0
            
            while not done and steps < self.config.environment.max_steps:
                # 选择动作
                actions, log_probs, values = self.select_action(obs, training=True)
                
                # 记录价值函数（用于调整通信频率）
                self.last_values = [v.item() for v in values]
                
                # 执行动作
                next_obs, rewards, dones, _ = self.env.step(actions)
                
                # 存储经验
                self.store_transition(
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
                self.total_steps += 1
                
                # 统计通信
                if self.comm_history and self.comm_history[-1]['should_communicate']:
                    episode_comm_count += 1
                
                # 检查是否结束
                if isinstance(dones, list):
                    done = all(dones)
                else:
                    done = dones
            
            # 记录
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            comm_rate = episode_comm_count / steps if steps > 0 else 0
            comm_rates.append(comm_rate)
            self.total_episodes += 1
            
            # 打印进度
            if (episode + 1) % self.config.training.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.training.log_interval:])
                avg_comm_rate = np.mean(comm_rates[-self.config.training.log_interval:])
                print(f"Episode {episode + 1}/{self.config.training.num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Comm Rate: {avg_comm_rate:.2%}, "
                      f"Steps: {steps}")
            
            # 保存最佳模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model(f"models/iacn_best_{self.config.experiment.env}.pt")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'comm_rates': comm_rates,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward
        }
    
    def get_communication_stats(self):
        """获取通信统计信息"""
        if not self.comm_history:
            return {
                'communication_rate': 0.0,
                'avg_message_size': self.obs_dim * 4 / 1024,  # KB
                'topology_type': self.comm_topology_type
            }
        
        # 计算通信率
        comm_steps = sum(1 for comm in self.comm_history if comm['should_communicate'])
        total_steps = len(self.comm_history)
        comm_rate = comm_steps / total_steps if total_steps > 0 else 0
        
        # 计算平均邻居数
        avg_neighbors = 0
        neighbor_counts = []
        for comm in self.comm_history:
            if comm['should_communicate'] and 'topology' in comm:
                topology = comm['topology']
                for i in topology:
                    neighbor_counts.append(len(topology[i]))
        
        avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
        
        return {
            'communication_rate': comm_rate,
            'avg_message_size': self.obs_dim * 4 / 1024,  # 假设传输完整观测，每个元素4字节
            'avg_neighbors': avg_neighbors,
            'topology_type': self.comm_topology_type
        }


class CommunicationNetwork(nn.Module):
    """通信网络：编码和解码消息"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # 初始化参数
        for layer in [self.encoder, self.decoder]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.zeros_(module.bias)
    
    def encode(self, x):
        """编码观测为消息"""
        return self.encoder(x)
    
    def decode(self, x):
        """解码消息为观测"""
        return self.decoder(x)
    
    def forward(self, x):
        """完整编码-解码过程"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded