#!/usr/bin/env python3
"""
多智能体强化学习基础算法类 - 修复版
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque


class BaseAlgorithm:
    """多智能体强化学习基础算法类"""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = config.device
        
        # 环境参数
        self.num_agents = env.num_agents
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        
        # 训练参数
        self.gamma = config.training.gamma
        self.gae_lambda = config.training.gae_lambda
        self.ppo_clip = config.training.ppo_clip
        self.value_coef = config.training.value_coef
        self.entropy_coef = config.training.entropy_coef
        self.max_grad_norm = config.training.max_grad_norm
        
        # 网络
        self.action_policies = nn.ModuleList([
            ActorNetwork(self.obs_dim, self.action_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)
        
        self.value_networks = nn.ModuleList([
            ValueNetwork(self.obs_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.action_policies.parameters()) +
            list(self.value_networks.parameters()),
            lr=config.training.learning_rate
        )
        
        # 经验缓冲区
        self.buffer_size = config.training.buffer_size
        self.batch_size = config.training.batch_size
        self.reset_buffer()
        
        # 训练状态
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')
        
        print(f"BaseAlgorithm initialized with {self.num_agents} agents")
    
    def reset_buffer(self):
        """重置经验缓冲区"""
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
    
    def store_transition(self, obs, actions, rewards, next_obs, dones, values, log_probs):
        """存储经验到缓冲区"""
        # 辅助函数：将数据移动到CPU
        def to_cpu(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, list):
                # 递归处理列表
                return [to_cpu(item) for item in data]
            else:
                return data
        
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(actions)
        self.buffer['rewards'].append(rewards)
        self.buffer['next_obs'].append(next_obs)
        self.buffer['dones'].append(dones)
        
        # 确保values和log_probs在CPU上
        self.buffer['values'].append(to_cpu(values))
        self.buffer['log_probs'].append(to_cpu(log_probs))
        
        # 如果缓冲区满了，进行训练
        if len(self.buffer['obs']) >= self.buffer_size:
            self.update()
    
    def select_action(self, obs, training=True):
        """选择动作（子类需要重写）"""
        raise NotImplementedError
    
    def train(self):
        """训练算法（子类需要重写）"""
        raise NotImplementedError
    
    def evaluate(self, num_episodes=10):
        """评估算法性能"""
        total_rewards = []
        success_count = 0
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.config.environment.max_steps:
                # 选择动作（评估模式下不探索）
                actions, _, _ = self.select_action(obs, training=False)
                
                # 执行动作
                next_obs, rewards, dones, _ = self.env.step(actions)
                
                # 更新
                obs = next_obs
                episode_reward += np.mean(rewards)
                steps += 1
                
                # 检查是否结束
                if isinstance(dones, list):
                    done = all(dones)
                else:
                    done = dones
            
            # 记录结果
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # 判断是否成功（简化：奖励大于阈值）
            if episode_reward > self.config.evaluation.success_threshold:
                success_count += 1
        
        # 计算统计信息
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        success_rate = success_count / num_episodes
        avg_length = np.mean(episode_lengths)
        
        # 获取通信统计（如果算法支持）
        comm_stats = self.get_communication_stats() if hasattr(self, 'get_communication_stats') else {}
        comm_cost = comm_stats.get('avg_message_size', 0.0) / 1024  # 转换为KB
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'avg_length': avg_length,
            'comm_cost': comm_cost,
            'total_rewards': total_rewards,
            'episode_lengths': episode_lengths
        }
    
    def update(self):
        """更新网络参数"""
        if len(self.buffer['obs']) < self.batch_size:
            return
        
        # 准备批量数据
        batch_indices = random.sample(range(len(self.buffer['obs'])), self.batch_size)
        
        batch_obs = [self.buffer['obs'][i] for i in batch_indices]
        batch_actions = [self.buffer['actions'][i] for i in batch_indices]
        batch_rewards = [self.buffer['rewards'][i] for i in batch_indices]
        batch_next_obs = [self.buffer['next_obs'][i] for i in batch_indices]
        batch_dones = [self.buffer['dones'][i] for i in batch_indices]
        batch_values = [self.buffer['values'][i] for i in batch_indices]
        batch_log_probs = [self.buffer['log_probs'][i] for i in batch_indices]
        
        # 对多智能体的rewards和values进行平均
        batch_rewards_processed = []
        batch_values_processed = []

        for r in batch_rewards:
            if isinstance(r, (list, np.ndarray)):
                batch_rewards_processed.append(np.mean(r))
            else:
                batch_rewards_processed.append(r)

        for v in batch_values:
            if isinstance(v, (list, np.ndarray)):
                batch_values_processed.append(np.mean(v))
            else:
                batch_values_processed.append(v)

        batch_rewards = batch_rewards_processed
        batch_values = batch_values_processed

        # 计算优势函数和回报
        advantages, returns = self._compute_advantages_and_returns(
            batch_rewards, batch_values, batch_dones
        )
        
        # 转换为张量
        batch_obs_tensor = torch.FloatTensor(np.array(batch_obs)).to(self.device)
        batch_actions_tensor = torch.FloatTensor(np.array(batch_actions)).to(self.device)
        
        # 确保log_probs是numpy数组
        batch_log_probs_array = np.array(batch_log_probs)

        # 如果log_probs是多维的（多智能体），取平均
        if batch_log_probs_array.ndim > 1:
            batch_log_probs_array = batch_log_probs_array.mean(axis=1)

        batch_log_probs_tensor = torch.FloatTensor(batch_log_probs_array).to(self.device)
        
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 计算损失
        losses = self.compute_losses({
            'obs': batch_obs_tensor,
            'actions': batch_actions_tensor,
            'old_log_probs': batch_log_probs_tensor,
            'advantages': advantages_tensor,
            'returns': returns_tensor
        })
        
        # 更新参数
        self.update_parameters(losses)
        
        # 清空缓冲区
        self.reset_buffer()
    
    def _compute_advantages_and_returns(self, rewards, values, dones):
        """计算优势函数和回报"""
        advantages = []
        returns = []
        
        for episode_rewards, episode_values, episode_dones in zip(rewards, values, dones):
            # 确保episode_rewards是列表或数组
            if not isinstance(episode_rewards, (list, np.ndarray)):
                # 如果是标量值，包装在列表中
                episode_rewards = [episode_rewards]

            # 确保episode_values是numpy数组
            if isinstance(episode_values, torch.Tensor):
                episode_values = episode_values.detach().cpu().numpy()
            elif isinstance(episode_values, list):
                # 如果values是列表，转换为numpy数组
                episode_values = np.array(episode_values)

            # 如果values是多维的（多智能体），取平均
            if episode_values.ndim > 1:
                episode_values = episode_values.mean(axis=1)
            
            # 确保episode_values也是列表或数组
            if not isinstance(episode_values, (list, np.ndarray)):
                episode_values = [episode_values]

            episode_length = len(episode_rewards)
            episode_advantages = np.zeros(episode_length)
            episode_returns = np.zeros(episode_length)
            
            last_advantage = 0
            last_return = 0
            
            # 反向计算
            for t in reversed(range(episode_length)):
                if t == episode_length - 1 or episode_dones[t]:
                    delta = episode_rewards[t] - episode_values[t]
                    last_advantage = delta
                else:
                    next_value = episode_values[t + 1] if t + 1 < episode_length else 0
                    delta = episode_rewards[t] + self.gamma * next_value - episode_values[t]
                    
                episode_advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
                episode_returns[t] = episode_advantages[t] + episode_values[t]
            
            advantages.extend(episode_advantages)
            returns.extend(episode_returns)
        
        return np.array(advantages), np.array(returns)
    
    def compute_losses(self, batch):
        """计算损失函数"""
        # 重新计算动作分布和值函数
        new_log_probs = []
        new_values = []
        entropies = []
        
        for i in range(self.num_agents):
            # 动作分布
            action_dist = self.action_policies[i](batch['obs'][:, i])
            new_log_prob = action_dist.log_prob(batch['actions'][:, i])
            entropy = action_dist.entropy()
            
            # 值函数
            value = self.value_networks[i](batch['obs'][:, i])
            
            new_log_probs.append(new_log_prob)
            new_values.append(value)
            entropies.append(entropy)
        
        # 平均所有智能体的损失
        new_log_probs = torch.stack(new_log_probs).mean(dim=0)
        new_values = torch.stack(new_values).mean(dim=0)
        entropy = torch.stack(entropies).mean()

        # 如果log_probs是多维的（多元正态分布），对动作维度求和
        if new_log_probs.ndim > 1:
            new_log_probs = new_log_probs.sum(dim=-1)
        
        # PPO损失
        ratio = torch.exp(new_log_probs - batch['old_log_probs'])
        surr1 = ratio * batch['advantages']
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * batch['advantages']
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 值函数损失
        value_loss = F.mse_loss(new_values, batch['returns'])
        
        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
    
    def update_parameters(self, losses):
        """更新网络参数"""
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.action_policies.parameters()) +
            list(self.value_networks.parameters()),
            max_norm=self.max_grad_norm
        )
        self.optimizer.step()
    
    def save_model(self, path):
        """保存模型"""
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'action_policies': self.action_policies.state_dict(),
            'value_networks': self.value_networks.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.action_policies.load_state_dict(checkpoint['action_policies'])
        self.value_networks.load_state_dict(checkpoint['value_networks'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class ActorNetwork(nn.Module):
    """行动策略网络"""
    
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)
        
        # 初始化参数
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_std.weight, gain=0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-4  # 确保标准差为正
        
        return torch.distributions.Normal(mean, std)


class ValueNetwork(nn.Module):
    """值函数网络"""
    
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # 初始化参数
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value