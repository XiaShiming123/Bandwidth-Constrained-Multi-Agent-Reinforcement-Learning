#!/usr/bin/env python3
"""
MAPPO (Multi-Agent PPO) 算法实现 - 最终修复版
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from experiments.algorithms.base_algorithm import BaseAlgorithm


class MAPPO(BaseAlgorithm):
    """MAPPO算法：集中式critic，分布式actor"""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # 集中式critic网络
        self.central_critic = CentralizedCritic(
            self.state_dim,
            self.config.mappo.critic_hidden_dim
        ).to(self.device)
        
        # 优化器
        self.critic_optimizer = torch.optim.Adam(
            self.central_critic.parameters(),
            lr=config.training.learning_rate
        )
        
        print("MAPPO initialized with centralized critic")
    
    def select_action(self, obs, training=True):
        """选择动作"""
        # 将观测转换为numpy数组，然后转换为张量
        if isinstance(obs, list):
            obs_array = np.array(obs)
        else:
            obs_array = obs
        
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        
        actions = []
        action_log_probs = []
        values = []
        
        # 分布式行动决策
        for i in range(self.num_agents):
            action_dist = self.action_policies[i](obs_tensor[i])
            
            if training:
                action = action_dist.sample()
            else:
                action = action_dist.mean
                
            action_log_prob = action_dist.log_prob(action)
            
            # 确保数据在CPU上
            actions.append(action.detach().cpu().numpy())
            action_log_probs.append(action_log_prob.detach().cpu().numpy())
        
        # 集中式值函数估计
        state = self._get_global_state(obs)
        state_tensor = torch.FloatTensor(state).to(self.device)
        value = self.central_critic(state_tensor)
        
        # 为每个智能体复制相同的值，并确保在CPU上
        value_cpu = value.detach().cpu().numpy()
        values = [value_cpu] * self.num_agents
        
        # 计算所有智能体的平均log_prob
        avg_log_prob = np.mean(action_log_probs)

        return actions, avg_log_prob, values
    
    def _get_global_state(self, obs):
        """获取全局状态"""
        if isinstance(obs, list):
            return np.concatenate(obs)
        else:
            return obs.flatten()
    
    def compute_losses(self, batch):
        """计算损失函数"""
        # 基础PPO损失
        policy_loss, value_loss = self._compute_ppo_loss(batch)
        
        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def _compute_ppo_loss(self, batch):
        """计算PPO损失"""
        # 重新计算动作分布
        new_log_probs = []
        entropies = []
        
        for i in range(self.num_agents):
            action_dist = self.action_policies[i](batch['obs'][:, i])
            new_log_prob = action_dist.log_prob(batch['actions'][:, i])
            entropy = action_dist.entropy()
            
            new_log_probs.append(new_log_prob)
            entropies.append(entropy)
        
        # 平均所有智能体的损失
        # new_log_probs 是一个列表，每个元素是一个智能体的 log_prob，形状为 [batch_size, action_dim]
        # 我们需要先对每个智能体的 log_prob 在 action_dim 上取平均，然后对所有智能体的 log_prob 取平均
        new_log_probs = torch.stack(new_log_probs)  # [num_agents, batch_size, action_dim]
        new_log_probs = new_log_probs.mean(dim=2)  # [num_agents, batch_size]
        new_log_probs = new_log_probs.mean(dim=0)  # [batch_size]
        entropy = torch.stack(entropies).mean()
        
        # 确保old_log_probs有正确的形状
        # batch['old_log_probs']可能是(batch_size,)或(batch_size, num_agents)
        old_log_probs = batch['old_log_probs']

        # 打印形状用于调试
        # print(f"new_log_probs shape: {new_log_probs.shape}, old_log_probs shape: {old_log_probs.shape}")

        if old_log_probs.dim() > 1:
            # 如果是(batch_size, num_agents)，取平均
            old_log_probs = old_log_probs.mean(dim=1)

        # 确保形状一致
        if old_log_probs.shape != new_log_probs.shape:
            old_log_probs = old_log_probs.reshape(new_log_probs.shape)
        
        # PPO策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * batch['advantages']
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * batch['advantages']
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 值函数损失
        # 重新计算全局状态的值
        batch_size = batch['obs'].size(0)
        batch_states = []
        for b in range(batch_size):
            state = self._get_global_state(batch['obs'][b].cpu().numpy())
            batch_states.append(state)
        
        batch_states_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
        new_values = self.central_critic(batch_states_tensor)
        
        value_loss = F.mse_loss(new_values.squeeze(), batch['returns'])
        
        # 添加熵正则化
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        return policy_loss, value_loss
    
    def update_parameters(self, losses):
        """更新参数"""
        # 更新actor网络
        self.optimizer.zero_grad()
        losses['policy_loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            self.action_policies.parameters(),
            max_norm=self.max_grad_norm
        )
        self.optimizer.step()
        
        # 更新critic网络
        self.critic_optimizer.zero_grad()
        losses['value_loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            self.central_critic.parameters(),
            max_norm=self.max_grad_norm
        )
        self.critic_optimizer.step()
    
    def train(self):
        """训练算法"""
        print(f"Training MAPPO for {self.config.training.num_episodes} episodes...")
        
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
                # 确保输出目录存在
                import os
                output_dir = self.config.experiment.output_dir
                model_dir = os.path.join(output_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"mappo_best_{self.config.experiment.env}.pt")
                self.save_model(model_path)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward
        }


class CentralizedCritic(nn.Module):
    """集中式critic网络"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化参数
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)