#!/usr/bin/env python3
"""
SparseComm 算法实现：固定稀疏通信拓扑
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from experiments.algorithms.base_algorithm import BaseAlgorithm


class SparseComm(BaseAlgorithm):
    """SparseComm算法：固定稀疏通信拓扑"""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # SparseComm特定参数
        self.k_neighbors = config.sparse_comm.k_neighbors
        self.comm_frequency = config.sparse_comm.comm_frequency  # 每N步通信一次
        
        # 通信网络
        self.comm_networks = nn.ModuleList([
            SparseCommunicationNetwork(self.obs_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)
        
        # 构建固定稀疏拓扑
        self.comm_topology = self._build_sparse_topology()
        
        # 通信历史
        self.comm_history = []
        self.comm_counter = 0
        
        print(f"SparseComm initialized with {self.k_neighbors}-nearest topology")
        print(f"Communication frequency: every {self.comm_frequency} steps")
    
    def _build_sparse_topology(self):
        """构建固定稀疏拓扑（最近邻）"""
        topology = {}
        
        # 获取智能体初始位置
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
        
        # 为每个智能体选择最近的k个邻居
        k = min(self.k_neighbors, self.num_agents - 1)
        for i in range(self.num_agents):
            distances = [(j, dist_matrix[i, j]) for j in range(self.num_agents) if j != i]
            distances.sort(key=lambda x: x[1])
            topology[i] = [j for j, _ in distances[:k]]
        
        return topology
    
    def select_action(self, obs, training=True):
        """选择动作，包括固定频率通信"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # 决定是否通信（基于固定频率）
        should_communicate = self._should_communicate()
        
        # 如果通信，交换信息
        if should_communicate:
            updated_obs = self._sparse_communicate(obs_tensor)
            
            # 记录通信
            self.comm_history.append({
                'step': self.total_steps,
                'should_communicate': True,
                'topology': self.comm_topology,
                'k_neighbors': self.k_neighbors
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
        """决定是否进行通信（固定频率）"""
        self.comm_counter += 1
        return self.comm_counter % self.comm_frequency == 0
    
    def _sparse_communicate(self, obs_tensor):
        """执行稀疏通信"""
        updated_obs = obs_tensor.clone()
        
        # 每个智能体向稀疏拓扑中的邻居发送消息
        for i in range(self.num_agents):
            if i in self.comm_topology:
                neighbors = self.comm_topology[i]
                if neighbors:
                    # 编码观测（使用稀疏编码）
                    encoded_message = self.comm_networks[i].encode_sparse(obs_tensor[i])
                    
                    # 向每个邻居发送消息
                    for neighbor in neighbors:
                        # 解码消息
                        decoded_message = self.comm_networks[neighbor].decode_sparse(encoded_message)
                        
                        # 整合到邻居的观测中
                        updated_obs[neighbor] = updated_obs[neighbor] + 0.1 * decoded_message
        
        return updated_obs
    
    def train(self):
        """训练算法"""
        print(f"Training SparseComm for {self.config.training.num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        comm_rates = []
        
        for episode in range(self.config.training.num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            episode_comm_count = 0
            
            # 重置通信计数器
            self.comm_counter = 0
            
            while not done and steps < self.config.environment.max_steps:
                # 选择动作
                actions, log_probs, values = self.select_action(obs, training=True)
                
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
                self.save_model(f"models/sparse_comm_best_{self.config.experiment.env}.pt")
        
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
                'k_neighbors': self.k_neighbors,
                'comm_frequency': self.comm_frequency
            }
        
        # 计算通信率
        comm_steps = sum(1 for comm in self.comm_history if comm['should_communicate'])
        total_steps = len(self.comm_history)
        comm_rate = comm_steps / total_steps if total_steps > 0 else 0
        
        # 计算平均消息大小（稀疏编码）
        # 稀疏编码通常传输原始观测的30%
        sparse_factor = 0.3
        avg_message_size = self.obs_dim * 4 * sparse_factor / 1024  # KB
        
        return {
            'communication_rate': comm_rate,
            'avg_message_size': avg_message_size,
            'k_neighbors': self.k_neighbors,
            'comm_frequency': self.comm_frequency,
            'topology_type': f'{self.k_neighbors}-nearest'
        }


class SparseCommunicationNetwork(nn.Module):
    """稀疏通信网络"""
    
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        
        # 稀疏编码器：选择最重要的特征
        self.importance_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
            nn.Sigmoid()  # 输出重要性分数
        )
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        
        # 初始化参数
        for net in [self.importance_net, self.encoder, self.decoder]:
            for module in net:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.zeros_(module.bias)
    
    def encode_sparse(self, x):
        """稀疏编码：选择重要特征进行编码"""
        # 计算重要性分数
        importance = self.importance_net(x)
        
        # 选择最重要的30%特征
        k = int(self.obs_dim * 0.3)
        _, indices = torch.topk(importance, k)
        
        # 创建稀疏向量
        sparse_x = torch.zeros_like(x)
        sparse_x[indices] = x[indices]
        
        # 编码稀疏向量
        encoded = self.encoder(sparse_x)
        
        return {
            'encoded': encoded,
            'indices': indices,
            'importance': importance
        }
    
    def decode_sparse(self, sparse_data):
        """解码稀疏消息"""
        encoded = sparse_data['encoded']
        indices = sparse_data['indices']
        
        # 解码
        decoded_full = self.decoder(encoded)
        
        # 创建完整向量（只有重要位置有值）
        result = torch.zeros(self.obs_dim, device=encoded.device)
        result[indices] = decoded_full[:len(indices)]
        
        return result
    
    def forward(self, x):
        """完整编码-解码过程"""
        sparse_data = self.encode_sparse(x)
        decoded = self.decode_sparse(sparse_data)
        return decoded