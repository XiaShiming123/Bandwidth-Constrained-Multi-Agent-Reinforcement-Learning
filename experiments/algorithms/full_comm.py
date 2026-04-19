
#!/usr/bin/env python3
"""
Full-Comm 算法实现：无约束完全通信
作为理论上限基准
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from experiments.algorithms.base_algorithm import BaseAlgorithm


class FullComm(BaseAlgorithm):
    """Full-Comm算法：无约束完全通信"""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # Full-Comm特定设置
        self.comm_every_step = config.full_comm.comm_every_step
        
        # 通信网络（全连接，无压缩）
        self.comm_networks = nn.ModuleList([
            FullCommunicationNetwork(self.obs_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)
        
        # 通信历史
        self.comm_history = []
        
        print("FullComm initialized with full communication every step")
        print("Note: This serves as the theoretical upper bound")
    
    def select_action(self, obs, training=True):
        """选择动作，每步都进行完全通信"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # 每步都进行完全通信
        updated_obs = self._full_communicate(obs_tensor)
        
        # 记录通信
        self.comm_history.append({
            'step': self.total_steps,
            'should_communicate': True,
            'comm_type': 'full'
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
    
    def _full_communicate(self, obs_tensor):
        """执行完全通信：每个智能体与所有其他智能体通信"""
        updated_obs = obs_tensor.clone()
        
        # 收集所有智能体的观测
        all_obs = []
        for i in range(self.num_agents):
            # 编码观测（无压缩）
            encoded_obs = self.comm_networks[i].encode_full(obs_tensor[i])
            all_obs.append(encoded_obs)
        
        # 每个智能体接收所有其他智能体的信息
        for i in range(self.num_agents):
            # 收集其他智能体的信息
            other_obs = [all_obs[j] for j in range(self.num_agents) if j != i]
            
            if other_obs:
                # 平均其他智能体的信息
                combined_obs = torch.mean(torch.stack(other_obs), dim=0)
                
                # 解码并整合到观测中
                decoded_info = self.comm_networks[i].decode_full(combined_obs)
                
                # 完全整合（权重较高）
                updated_obs[i] = updated_obs[i] + 0.2 * decoded_info
        
        return updated_obs
    
    def train(self):
        """训练算法"""
        print(f"Training FullComm for {self.config.training.num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.training.num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
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
                
                # 检查是否结束
                if isinstance(dones, list):
                    done = all(dones)
                else:
                    done = dones
            
            # 记录
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            self.total_episodes += 1
            
            # 打印进度
            if (episode + 1) % self.config.training.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.training.log_interval:])
                print(f"Episode {episode + 1}/{self.config.training.num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Steps: {steps}")
            
            # 保存最佳模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model(f"models/full_comm_best_{self.config.experiment.env}.pt")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward
        }
    
    def get_communication_stats(self):
        """获取通信统计信息"""
        # Full-Comm每步都通信
        comm_rate = 1.0
        
        # 每个智能体与其他所有智能体通信
        avg_neighbors = self.num_agents - 1
        
        # 传输完整观测，无压缩
        avg_message_size = self.obs_dim * 4 / 1024  # KB
        
        # 总通信量（每个智能体每步）
        total_comm_per_step = avg_message_size * avg_neighbors * self.num_agents
        
        return {
            'communication_rate': comm_rate,
            'avg_message_size': avg_message_size,
            'avg_neighbors': avg_neighbors,
            'total_comm_per_step': total_comm_per_step,
            'comm_type': 'full_unconstrained'
        }


class FullCommunicationNetwork(nn.Module):
    """完全通信网络：无压缩编码"""
    
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        
        # 编码器（无压缩，保持维度）
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)  # 保持较大维度以保留信息
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )
        
        # 初始化参数
        for module in [self.encoder, self.decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
    
    def encode_full(self, x):
        """完全编码：无信息损失"""
        return self.encoder(x)
    
    def decode_full(self, x):
        """完全解码"""
        return self.decoder(x)
    
    def forward(self, x):
        """完整编码-解码过程"""
        encoded = self.encode_full(x)
        decoded = self.decode_full(encoded)
        return decoded