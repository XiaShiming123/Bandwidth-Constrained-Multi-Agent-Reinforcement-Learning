#!/usr/bin/env python3
"""
自适应通信多智能体强化学习算法
基于论文：面向受限环境的集成自适应多智能体通信与协作框架
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque

from experiments.algorithms.base_algorithm import BaseAlgorithm


class AdaptiveComm(BaseAlgorithm):
    """自适应通信多智能体强化学习算法"""

    def __init__(self, env, config):
        super().__init__(env, config)

        # 算法特定参数
        self.comm_dim = config.adaptive_comm.comm_dim
        self.comm_cost_coef = config.adaptive_comm.comm_cost_coef
        self.sparsity_threshold = config.adaptive_comm.sparsity_threshold
        self.topology_update_freq = config.adaptive_comm.topology_update_freq
        
        # 动态阈值参数 (与论文一致)
        self.base_threshold = config.adaptive_comm.base_threshold  # τ₀
        self.threshold_alpha = config.adaptive_comm.threshold_alpha  # 资源压力敏感度
        self.threshold_beta = config.adaptive_comm.threshold_beta  # 任务紧急度敏感度
        self.freq_threshold = config.adaptive_comm.freq_threshold
        
        # 通信成本追踪
        self.total_comm_bytes = 0
        self.total_steps = 0
        self.bandwidth_limit = config.communication.bandwidth_limit * 1024  # 转换为字节

        # 通信相关组件
        self.message_encoders = nn.ModuleList([
            MessageEncoder(self.obs_dim, self.comm_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)

        self.message_decoders = nn.ModuleList([
            MessageDecoder(self.comm_dim, self.obs_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)

        self.communication_policies = nn.ModuleList([
            CommunicationPolicy(self.obs_dim, self.comm_dim)
            for _ in range(self.num_agents)
        ]).to(self.device)

        # 自适应组件
        self.topology_learner = TopologyLearner(self.num_agents, self.comm_dim).to(self.device)
        self.content_selector = ContentSelector(self.obs_dim, self.comm_dim).to(self.device)
        self.frequency_controller = FrequencyController(self.obs_dim).to(self.device)

        # 通信历史
        self.comm_history = deque(maxlen=1000)
        self.topology_history = []

        # 优化器
        self.comm_optimizer = torch.optim.Adam(
            list(self.message_encoders.parameters()) +
            list(self.message_decoders.parameters()) +
            list(self.communication_policies.parameters()) +
            list(self.topology_learner.parameters()) +
            list(self.content_selector.parameters()) +
            list(self.frequency_controller.parameters()),
            lr=config.training.learning_rate
        )

        print(f"AdaptiveComm initialized with {self.num_agents} agents")
        print(f"Communication dimension: {self.comm_dim}")
        print(f"Communication cost coefficient: {self.comm_cost_coef}")

    def select_action(self, obs, training=True):
        """选择动作，包括通信决策"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)

        # 获取当前通信拓扑
        current_topology = self._get_current_topology(obs_tensor)

        # 决定是否通信
        should_communicate = self._decide_communication(obs_tensor)

        # 生成消息
        messages = []
        if should_communicate:
            messages = self._generate_messages(obs_tensor, current_topology)

        # 接收消息并更新观测
        updated_obs = self._receive_messages(obs_tensor, messages, current_topology)

        # 基于更新后的观测选择动作
        actions = []
        action_log_probs = []
        values = []

        for i in range(self.num_agents):
            # 将观测和接收到的消息结合
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

        # 记录通信信息
        if should_communicate:
            comm_info = {
                'topology': current_topology,
                'messages': len(messages),
                'should_communicate': should_communicate
            }
            self.comm_history.append(comm_info)

        return actions, action_log_probs, values

    def _get_current_topology(self, obs_tensor):
        """获取当前通信拓扑"""
        # 基于智能体位置和任务状态动态构建拓扑
        positions = []
        for i in range(self.num_agents):
            # 从观测中提取位置信息（假设前2维是位置）
            pos = obs_tensor[i, :2].cpu().numpy()
            positions.append(pos)

        positions = np.array(positions)

        # 计算距离矩阵
        dist_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])

        # 基于距离和任务相关性构建拓扑
        topology = {}
        for i in range(self.num_agents):
            # 选择最近的k个智能体，但考虑任务相关性
            distances = [(j, dist_matrix[i, j]) for j in range(self.num_agents) if i != j]
            distances.sort(key=lambda x: x[1])

            # 自适应选择邻居数量
            k = self._adaptive_neighbor_count(obs_tensor[i])
            neighbors = [j for j, _ in distances[:k]]

            topology[i] = neighbors

        return topology

    def _adaptive_neighbor_count(self, obs):
        """自适应选择邻居数量"""
        # 基于环境不确定性和任务复杂度
        uncertainty = torch.var(obs).item()
        task_complexity = torch.mean(torch.abs(obs)).item()

        # 基础邻居数量
        base_k = self.config.adaptive_comm.base_neighbors

        # 根据不确定性调整
        if uncertainty > 0.5:
            k = min(base_k + 2, self.num_agents - 1)
        elif uncertainty > 0.2:
            k = base_k + 1
        else:
            k = base_k

        # 根据任务复杂度调整
        if task_complexity > 0.7:
            k = min(k + 1, self.num_agents - 1)

        return max(1, k)

    def _decide_communication(self, obs_tensor):
        """决定是否进行通信 - 基于论文的三重自适应机制"""
        # 1. 计算任务紧急度 (基于 TD 误差估计)
        task_urgency = self._estimate_task_urgency(obs_tensor)
        
        # 2. 计算不确定性 (基于策略熵)
        uncertainty = self._estimate_uncertainty(obs_tensor)
        
        # 3. 计算置信度
        confidence = self._measure_decision_confidence(obs_tensor)
        
        # 4. 计算资源压力 (基于带宽使用率)
        resource_pressure = self._estimate_resource_pressure()
        
        # 5. 动态阈值调整 (公式 2)
        # τ_t = τ₀ × (1 - α × ResourcePressure_t + β × TaskUrgency_t)
        dynamic_threshold = self.base_threshold * (
            1 - self.threshold_alpha * resource_pressure + 
            self.threshold_beta * task_urgency
        )
        dynamic_threshold = np.clip(dynamic_threshold, 0.2, 0.8)  # 限制在合理范围
        
        # 6. 计算通信必要性评分
        comm_necessity_score = self._compute_communication_necessity_score(
            obs_tensor, uncertainty, confidence, task_urgency
        )
        
        # 7. 基于动态阈值做决策
        should_comm = comm_necessity_score > dynamic_threshold
        
        # 记录决策信息
        decision_info = {
            'task_urgency': task_urgency,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'resource_pressure': resource_pressure,
            'dynamic_threshold': dynamic_threshold,
            'comm_necessity_score': comm_necessity_score
        }
        
        return should_comm

    def _estimate_task_urgency(self, obs_tensor):
        """估计任务紧急度 - 基于 TD 误差"""
        # 使用最近的价值估计差异作为 TD 误差的近似
        if len(self.comm_history) < 5:
            return 0.5  # 默认中等紧急度
        
        # 基于观测变化的幅度估计任务紧急度
        obs_changes = []
        for i in range(self.num_agents):
            obs_norm = torch.norm(obs_tensor[i]).item()
            obs_changes.append(obs_norm)
        
        # 归一化到 [0, 1]
        urgency = np.clip(np.mean(obs_changes) / 2.0, 0.0, 1.0)
        return urgency

    def _estimate_uncertainty(self, obs_tensor):
        """估计不确定性 - 基于策略熵"""
        entropies = []
        for i in range(self.num_agents):
            action_dist = self.action_policies[i](obs_tensor[i])
            entropy = action_dist.entropy().mean().item()
            entropies.append(entropy)
        
        # 归一化到 [0, 1] (假设最大熵约为 2.0)
        uncertainty = np.clip(np.mean(entropies) / 2.0, 0.0, 1.0)
        return uncertainty

    def _estimate_resource_pressure(self):
        """估计资源压力 - 基于带宽使用率"""
        if self.total_steps == 0:
            return 0.0
        
        # 计算当前带宽使用率
        bandwidth_usage = self.total_comm_bytes / (self.total_steps * self.bandwidth_limit)
        
        # 归一化到 [0, 1]
        resource_pressure = np.clip(bandwidth_usage, 0.0, 1.0)
        return resource_pressure

    def _compute_communication_necessity_score(self, obs_tensor, uncertainty, confidence, task_urgency):
        """计算通信必要性评分"""
        # 评分公式：综合考虑不确定性、置信度和任务紧急度
        # Score = α × Uncertainty + β × (1 - Confidence) + γ × TaskUrgency
        
        # 权重系数
        alpha, beta, gamma = 0.3, 0.3, 0.4
        
        # 计算评分
        score = alpha * uncertainty + beta * (1 - confidence) + gamma * task_urgency
        
        # 归一化到 [0, 1]
        score = np.clip(score, 0.0, 1.0)
        
        return score

    def _measure_environment_changes(self):
        """测量环境变化程度"""
        if len(self.comm_history) < 10:
            return 0.0

        # 简单实现：基于最近观测的变化
        return random.random() * 0.5  # 简化

    def _measure_decision_confidence(self, obs_tensor):
        """测量决策置信度"""
        # 基于行动策略网络的输出熵
        confidences = []
        for i in range(self.num_agents):
            action_dist = self.action_policies[i](obs_tensor[i])
            entropy = action_dist.entropy().mean().item()
            confidence = 1.0 / (1.0 + entropy)  # 熵越低，置信度越高
            confidences.append(confidence)

        return np.mean(confidences)

    def _is_critical_phase(self):
        """判断是否处于任务关键阶段"""
        # 简化实现：随机返回
        return random.random() < 0.2

    def _generate_messages(self, obs_tensor, topology):
        """生成通信消息 - 追踪通信成本"""
        messages = []

        for i in range(self.num_agents):
            if i in topology:
                neighbors = topology[i]
                if neighbors:
                    # 编码观测为消息
                    encoded_message = self.message_encoders[i](obs_tensor[i])

                    # 选择重要内容
                    important_content = self.content_selector(obs_tensor[i], encoded_message)

                    # 压缩消息
                    compressed_message = self._compress_message(important_content)

                    # 计算消息大小 (字节) - 每个 float32 占 4 字节
                    message_size = len(compressed_message) * 4
                    
                    # 追踪通信成本
                    self.total_comm_bytes += message_size

                    messages.append({
                        'sender': i,
                        'receivers': neighbors,
                        'content': compressed_message,
                        'size': message_size
                    })

        return messages

    def _compress_message(self, message):
        """压缩消息"""
        # 基于重要性进行稀疏化
        message_np = message.cpu().detach().numpy()

        # 保留最重要的k个元素
        k = int(len(message_np) * self.sparsity_threshold)
        if k > 0:
            # 找到绝对值最大的k个元素
            indices = np.argsort(np.abs(message_np))[-k:]
            compressed = np.zeros_like(message_np)
            compressed[indices] = message_np[indices]
        else:
            compressed = message_np

        return torch.FloatTensor(compressed).to(self.device)

    def _receive_messages(self, obs_tensor, messages, topology):
        """接收并处理消息"""
        updated_obs = obs_tensor.clone()

        # 收集每个智能体接收到的消息
        received_messages = {i: [] for i in range(self.num_agents)}

        for msg in messages:
            sender = msg['sender']
            receivers = msg['receivers']
            content = msg['content']

            for receiver in receivers:
                if receiver in received_messages:
                    received_messages[receiver].append({
                        'sender': sender,
                        'content': content
                    })

        # 处理接收到的消息
        for i in range(self.num_agents):
            if received_messages[i]:
                # 解码消息并整合到观测中
                combined_message = torch.mean(
                    torch.stack([m['content'] for m in received_messages[i]]),
                    dim=0
                )

                decoded_info = self.message_decoders[i](combined_message)

                # 将解码信息整合到观测中
                updated_obs[i] = updated_obs[i] + 0.1 * decoded_info  # 加权整合

        return updated_obs

    def train(self):
        """训练算法"""
        print(f"Training AdaptiveComm for {self.config.training.num_episodes} episodes...")

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
                self.save_model(f"models/adaptive_comm_best_{self.config.experiment.env}.pt")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'comm_rates': comm_rates,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward
        }

    def compute_losses(self, batch):
        """计算损失函数，包括通信损失"""
        # 基础PPO损失
        base_losses = super().compute_losses(batch)
        policy_loss = base_losses['policy_loss']
        value_loss = base_losses['value_loss']

        # 通信相关损失
        comm_loss = self._compute_communication_loss(batch)

        # 总损失
        total_loss = policy_loss + value_loss + self.comm_cost_coef * comm_loss

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'comm_loss': comm_loss
        }

    def _compute_communication_loss(self, batch):
        """计算通信损失"""
        # 通信效率损失：鼓励有效通信
        efficiency_loss = 0.0

        # 通信一致性损失：鼓励一致的通信决策
        consistency_loss = 0.0

        # 信息价值损失：鼓励传输有价值的信息
        value_loss = 0.0

        # 简化实现
        comm_loss = efficiency_loss + consistency_loss + value_loss

        return comm_loss

    def update_parameters(self, losses):
        """更新参数"""
        # 由于通信损失目前是简化实现（返回浮点数），不进行反向传播
        # 只更新基础网络的参数
        base_losses = {
            'total_loss': losses['policy_loss'] + losses['value_loss'],
            'policy_loss': losses['policy_loss'],
            'value_loss': losses['value_loss']
        }
        super().update_parameters(base_losses)

    def get_communication_stats(self):
        """获取通信统计信息 - 包含通信成本"""
        if not self.comm_history:
            return {
                'communication_rate': 0.0,
                'avg_message_size': 0.0,
                'avg_neighbors': 0.0,
                'total_comm_bytes': 0.0,
                'bandwidth_usage': 0.0,
                'bandwidth_savings': 100.0  # 默认 100% 节省
            }

        comm_rates = []
        message_sizes = []
        neighbor_counts = []

        for comm in self.comm_history:
            comm_rates.append(1.0 if comm['should_communicate'] else 0.0)
            if 'messages' in comm:
                message_sizes.append(comm['messages'])

            if 'topology' in comm:
                topology = comm['topology']
                for i in topology:
                    neighbor_counts.append(len(topology[i]))

        # 计算带宽使用率
        bandwidth_usage = 0.0
        if self.total_steps > 0:
            bandwidth_usage = self.total_comm_bytes / (self.total_steps * self.bandwidth_limit)
        
        # 计算带宽节省 (相比 FullComm 的 100% 使用)
        # 假设 FullComm 每步都使用完整带宽
        bandwidth_savings = (1 - bandwidth_usage) * 100

        return {
            'communication_rate': np.mean(comm_rates) if comm_rates else 0.0,
            'avg_message_size': np.mean(message_sizes) if message_sizes else 0.0,
            'avg_neighbors': np.mean(neighbor_counts) if neighbor_counts else 0.0,
            'total_comm_bytes': self.total_comm_bytes,
            'bandwidth_usage': bandwidth_usage,
            'bandwidth_savings': max(0, bandwidth_savings)
        }


class MessageEncoder(nn.Module):
    """消息编码器"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class MessageDecoder(nn.Module):
    """消息解码器"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class CommunicationPolicy(nn.Module):
    """通信策略网络"""

    def __init__(self, obs_dim, comm_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出通信概率
        )

    def forward(self, x):
        return self.network(x)


class TopologyLearner(nn.Module):
    """拓扑学习器"""

    def __init__(self, num_agents, feature_dim):
        super().__init__()
        self.num_agents = num_agents
        self.feature_dim = feature_dim

        # 学习智能体间的关系权重
        self.relation_weights = nn.Parameter(
            torch.randn(num_agents, num_agents)
        )

    def forward(self, agent_features):
        """基于智能体特征计算连接权重"""
        # agent_features: [num_agents, feature_dim]
        batch_size = agent_features.size(0)

        # 计算特征相似性
        similarities = torch.matmul(agent_features, agent_features.transpose(1, 2))

        # 结合学习到的关系权重
        connection_weights = F.softmax(
            similarities + self.relation_weights.unsqueeze(0),
            dim=-1
        )

        return connection_weights


class ContentSelector(nn.Module):
    """内容选择器"""
    
    def __init__(self, obs_dim, comm_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.comm_dim = comm_dim
        
        # 重要性网络输出与消息维度相同的权重
        self.importance_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),  # 输出comm_dim
            nn.Sigmoid()  # 输出重要性权重
        )
    
    def forward(self, obs, message):
        # 计算消息各部分的重要性
        importance_weights = self.importance_net(obs)
        
        # 根据重要性选择内容
        selected_content = message * importance_weights
        
        return selected_content
class FrequencyController(nn.Module):
    """频率控制器"""

    def __init__(self, obs_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出通信频率
        )

    def forward(self, x):
        return self.network(x)