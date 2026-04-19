#!/usr/bin/env python3
"""
配置文件加载工具
"""

import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass, field
from types import SimpleNamespace


def load_config(config_path: str) -> SimpleNamespace:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        SimpleNamespace: 配置对象
    """
    # 如果路径不存在，使用默认配置
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using default settings")
        return get_default_config()
    
    # 加载YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 转换为SimpleNamespace以便点访问
    config = dict_to_namespace(config_dict)
    
    # 设置默认值（如果缺失）
    config = set_default_values(config)
    
    return config


def dict_to_namespace(d: Dict) -> SimpleNamespace:
    """
    递归将字典转换为SimpleNamespace
    """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def set_default_values(config: SimpleNamespace) -> SimpleNamespace:
    """
    设置默认配置值
    """
    # 实验设置
    if not hasattr(config, 'experiment'):
        config.experiment = SimpleNamespace()
    if not hasattr(config.experiment, 'env'):
        config.experiment.env = "MPE_Navigation"
    if not hasattr(config.experiment, 'output_dir'):
        config.experiment.output_dir = "results"
    if not hasattr(config.experiment, 'device'):
        config.experiment.device = "cpu"
    
    # 环境设置
    if not hasattr(config, 'environment'):
        config.environment = SimpleNamespace()
    if not hasattr(config.environment, 'max_steps'):
        config.environment.max_steps = 200
    if not hasattr(config.environment, 'step_mul'):
        config.environment.step_mul = 8
    if not hasattr(config.environment, 'difficulty'):
        config.environment.difficulty = "7"
    if not hasattr(config.environment, 'replay_dir'):
        config.environment.replay_dir = "replays"
    
    # 通信约束
    if not hasattr(config, 'communication'):
        config.communication = SimpleNamespace()
    if not hasattr(config.communication, 'bandwidth_limit'):
        config.communication.bandwidth_limit = 10.0
    if not hasattr(config.communication, 'latency'):
        config.communication.latency = 1
    if not hasattr(config.communication, 'packet_loss'):
        config.communication.packet_loss = 0.1
    
    # 训练设置
    if not hasattr(config, 'training'):
        config.training = SimpleNamespace()
    if not hasattr(config.training, 'num_episodes'):
        config.training.num_episodes = 5000
    if not hasattr(config.training, 'max_steps'):
        config.training.max_steps = 200
    if not hasattr(config.training, 'buffer_size'):
        config.training.buffer_size = 2048
    if not hasattr(config.training, 'batch_size'):
        config.training.batch_size = 64
    if not hasattr(config.training, 'gamma'):
        config.training.gamma = 0.99
    if not hasattr(config.training, 'gae_lambda'):
        config.training.gae_lambda = 0.95
    if not hasattr(config.training, 'ppo_clip'):
        config.training.ppo_clip = 0.2
    if not hasattr(config.training, 'value_coef'):
        config.training.value_coef = 0.5
    if not hasattr(config.training, 'entropy_coef'):
        config.training.entropy_coef = 0.05
    if not hasattr(config.training, 'max_grad_norm'):
        config.training.max_grad_norm = 0.5
    if not hasattr(config.training, 'learning_rate'):
        config.training.learning_rate = 5e-4
    if not hasattr(config.training, 'log_interval'):
        config.training.log_interval = 10
    
    # 评估设置
    if not hasattr(config, 'evaluation'):
        config.evaluation = SimpleNamespace()
    if not hasattr(config.evaluation, 'num_episodes'):
        config.evaluation.num_episodes = 10
    if not hasattr(config.evaluation, 'success_threshold'):
        config.evaluation.success_threshold = 0.0
    
    # 算法特定设置
    if not hasattr(config, 'mappo'):
        config.mappo = SimpleNamespace()
    if not hasattr(config.mappo, 'critic_hidden_dim'):
        config.mappo.critic_hidden_dim = 128
    
    if not hasattr(config, 'iacn'):
        config.iacn = SimpleNamespace()
    if not hasattr(config.iacn, 'comm_freq_base'):
        config.iacn.comm_freq_base = 0.3
    if not hasattr(config.iacn, 'comm_freq_var'):
        config.iacn.comm_freq_var = 0.5
    if not hasattr(config.iacn, 'topology_type'):
        config.iacn.topology_type = "nearest"
    
    if not hasattr(config, 'sparse_comm'):
        config.sparse_comm = SimpleNamespace()
    if not hasattr(config.sparse_comm, 'k_neighbors'):
        config.sparse_comm.k_neighbors = 2
    if not hasattr(config.sparse_comm, 'comm_frequency'):
        config.sparse_comm.comm_frequency = 3
    
    if not hasattr(config, 'full_comm'):
        config.full_comm = SimpleNamespace()
    if not hasattr(config.full_comm, 'comm_every_step'):
        config.full_comm.comm_every_step = True
    
    if not hasattr(config, 'adaptive_comm'):
        config.adaptive_comm = SimpleNamespace()
    if not hasattr(config.adaptive_comm, 'comm_dim'):
        config.adaptive_comm.comm_dim = 8
    if not hasattr(config.adaptive_comm, 'comm_cost_coef'):
        config.adaptive_comm.comm_cost_coef = 0.1
    if not hasattr(config.adaptive_comm, 'sparsity_threshold'):
        config.adaptive_comm.sparsity_threshold = 0.3
    if not hasattr(config.adaptive_comm, 'topology_update_freq'):
        config.adaptive_comm.topology_update_freq = 10
    if not hasattr(config.adaptive_comm, 'base_neighbors'):
        config.adaptive_comm.base_neighbors = 2
    # 动态阈值参数 (与论文一致)
    if not hasattr(config.adaptive_comm, 'base_threshold'):
        config.adaptive_comm.base_threshold = 0.5
    if not hasattr(config.adaptive_comm, 'threshold_alpha'):
        config.adaptive_comm.threshold_alpha = 0.3
    if not hasattr(config.adaptive_comm, 'threshold_beta'):
        config.adaptive_comm.threshold_beta = 0.3
    if not hasattr(config.adaptive_comm, 'freq_threshold'):
        config.adaptive_comm.freq_threshold = 0.5
    
    # 日志设置
    if not hasattr(config, 'logging'):
        config.logging = SimpleNamespace()
    if not hasattr(config.logging, 'level'):
        config.logging.level = "INFO"
    if not hasattr(config.logging, 'format'):
        config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if not hasattr(config.logging, 'file'):
        config.logging.file = "experiment.log"
    
    # 可视化设置
    if not hasattr(config, 'visualization'):
        config.visualization = SimpleNamespace()
    if not hasattr(config.visualization, 'save_figures'):
        config.visualization.save_figures = True
    if not hasattr(config.visualization, 'figure_format'):
        config.visualization.figure_format = "png"
    if not hasattr(config.visualization, 'dpi'):
        config.visualization.dpi = 300
    
    return config


def get_default_config() -> SimpleNamespace:
    """
    获取默认配置
    """
    default_config = {
        'experiment': {
            'env': 'MPE_Navigation',
            'output_dir': 'results',
            'device': 'cpu'
        },
        'environment': {
            'max_steps': 200,
            'step_mul': 8,
            'difficulty': '7',
            'replay_dir': 'replays'
        },
        'communication': {
            'bandwidth_limit': 10.0,
            'latency': 1,
            'packet_loss': 0.1
        },
        'training': {
            'num_episodes': 5000,
            'max_steps': 200,
            'buffer_size': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'ppo_clip': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.05,
            'max_grad_norm': 0.5,
            'learning_rate': 5e-4,
            'log_interval': 10
        },
        'evaluation': {
            'num_episodes': 10,
            'success_threshold': 0.0
        },
        'mappo': {
            'critic_hidden_dim': 128
        },
        'iacn': {
            'comm_freq_base': 0.3,
            'comm_freq_var': 0.5,
            'topology_type': 'nearest'
        },
        'sparse_comm': {
            'k_neighbors': 2,
            'comm_frequency': 3
        },
        'full_comm': {
            'comm_every_step': True
        },
        'adaptive_comm': {
            'comm_dim': 8,
            'comm_cost_coef': 0.1,
            'sparsity_threshold': 0.3,
            'topology_update_freq': 10,
            'base_neighbors': 2,
            'base_threshold': 0.5,
            'threshold_alpha': 0.3,
            'threshold_beta': 0.3,
            'freq_threshold': 0.5
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'experiment.log'
        },
        'visualization': {
            'save_figures': True,
            'figure_format': 'png',
            'dpi': 300
        }
    }
    
    return dict_to_namespace(default_config)


def save_config(config: SimpleNamespace, path: str):
    """
    保存配置到文件
    """
    # 将SimpleNamespace转换为字典
    config_dict = namespace_to_dict(config)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存为YAML
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def namespace_to_dict(ns: SimpleNamespace) -> Dict:
    """
    递归将SimpleNamespace转换为字典
    """
    if isinstance(ns, SimpleNamespace):
        ns = ns.__dict__
    
    if isinstance(ns, dict):
        result = {}
        for key, value in ns.items():
            result[key] = namespace_to_dict(value)
        return result
    elif isinstance(ns, list):
        return [namespace_to_dict(item) for item in ns]
    else:
        return ns


def create_env_specific_config(env_name: str) -> SimpleNamespace:
    """
    创建环境特定的配置
    """
    config = get_default_config()
    config.experiment.env = env_name
    
    # 根据环境调整参数
    if env_name == "MPE_Navigation":
        config.environment.max_steps = 100
        config.training.num_episodes = 1000
        config.evaluation.success_threshold = 50.0
    elif env_name == "MPE_PredatorPrey":
        config.environment.max_steps = 150
        config.training.num_episodes = 1500
        config.evaluation.success_threshold = 80.0
    elif env_name == "SMAC_3m_vs_3z":
        config.environment.max_steps = 200
        config.training.num_episodes = 2000
        config.evaluation.success_threshold = 100.0
    
    return config


if __name__ == "__main__":
    # 测试配置加载
    config = load_config("configs/default.yaml")
    print(f"Environment: {config.experiment.env}")
    print(f"Training episodes: {config.training.num_episodes}")
    print(f"Learning rate: {config.training.learning_rate}")