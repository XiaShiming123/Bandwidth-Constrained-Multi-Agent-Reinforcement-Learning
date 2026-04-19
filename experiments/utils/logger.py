#!/usr/bin/env python3
"""
实验日志记录器
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, algorithm_name: str, env_name: str, seed: int, config):
        """初始化日志记录器"""
        self.algorithm_name = algorithm_name
        self.env_name = env_name
        self.seed = seed
        self.config = config

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{algorithm_name}_{env_name}_seed{seed}_{timestamp}"

        self.output_dir = os.path.join(
            config.experiment.output_dir,
            self.experiment_id
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化数据结构
        self.training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'comm_stats': []
        }

        self.evaluation_data = {
            'avg_reward': 0.0,
            'std_reward': 0.0,
            'success_rate': 0.0,
            'avg_length': 0.0,
            'comm_cost': 0.0,
            'total_rewards': [],
            'episode_lengths': []
        }

        # 设置日志
        self._setup_logging()

        # 保存配置
        self._save_config()

        self.logger.info(f"Experiment logger initialized: {self.experiment_id}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _setup_logging(self):
        """设置日志记录"""
        # 创建logger
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler
            log_file = os.path.join(self.output_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _save_config(self):
        """保存配置文件"""
        config_file = os.path.join(self.output_dir, "config.json")

        # 将SimpleNamespace转换为字典
        config_dict = self._namespace_to_dict(self.config)

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Configuration saved to: {config_file}")

    def _namespace_to_dict(self, ns):
        """将SimpleNamespace转换为字典"""
        if hasattr(ns, '__dict__'):
            ns = ns.__dict__

        if isinstance(ns, dict):
            result = {}
            for key, value in ns.items():
                result[key] = self._namespace_to_dict(value)
            return result
        elif isinstance(ns, list):
            return [self._namespace_to_dict(item) for item in ns]
        else:
            return ns

    def log_training(self, training_results: Dict[str, Any]):
        """记录训练结果"""
        self.training_data['episode_rewards'].extend(
            training_results.get('episode_rewards', [])
        )
        self.training_data['episode_lengths'].extend(
            training_results.get('episode_lengths', [])
        )

        # 记录损失（如果有）
        if 'losses' in training_results:
            self.training_data['losses'].append(training_results['losses'])

        # 记录通信统计（如果有）
        if 'comm_stats' in training_results:
            self.training_data['comm_stats'].append(training_results['comm_stats'])

        # 记录总步数和回合数
        self.total_steps = training_results.get('total_steps', 0)
        self.total_episodes = training_results.get('total_episodes', 0)
        self.best_reward = training_results.get('best_reward', -float('inf'))

        # 打印摘要
        if training_results.get('episode_rewards'):
            recent_rewards = training_results['episode_rewards'][-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            self.logger.info(
                f"Training - Episodes: {self.total_episodes}, "
                f"Steps: {self.total_steps}, "
                f"Recent Avg Reward: {avg_reward:.2f}, "
                f"Best Reward: {self.best_reward:.2f}"
            )

    def log_evaluation(self, evaluation_results: Dict[str, Any]):
        """记录评估结果"""
        self.evaluation_data.update(evaluation_results)

        self.logger.info(
            f"Evaluation - Avg Reward: {evaluation_results['avg_reward']:.2f} ± "
            f"{evaluation_results['std_reward']:.2f}, "
            f"Success Rate: {evaluation_results['success_rate']:.2%}, "
            f"Comm Cost: {evaluation_results['comm_cost']:.2f} KB/step"
        )

    def save(self):
        """保存所有数据到文件"""
        # 保存训练数据
        self._save_training_data()

        # 保存评估数据
        self._save_evaluation_data()

        # 保存摘要
        self._save_summary()

        self.logger.info(f"All data saved to: {self.output_dir}")

    def _save_training_data(self):
        """保存训练数据"""
        # CSV格式
        csv_file = os.path.join(self.output_dir, "training_data.csv")

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入标题
            writer.writerow([
                'episode', 'reward', 'length',
                'total_steps', 'best_reward'
            ])

            # 写入数据
            for i, (reward, length) in enumerate(zip(
                self.training_data['episode_rewards'],
                self.training_data['episode_lengths']
            )):
                writer.writerow([
                    i + 1, reward, length,
                    self.total_steps, self.best_reward
                ])

        # JSON格式（包含更多细节）
        json_file = os.path.join(self.output_dir, "training_data.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, default=str)

    def _save_evaluation_data(self):
        """保存评估数据"""
        json_file = os.path.join(self.output_dir, "evaluation_data.json")

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_data, f, indent=2, default=str)

    def _save_summary(self):
        """保存实验摘要"""
        summary = {
            'experiment_id': self.experiment_id,
            'algorithm': self.algorithm_name,
            'environment': self.env_name,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'best_reward': float(self.best_reward),
            'evaluation': {
                'avg_reward': float(self.evaluation_data['avg_reward']),
                'std_reward': float(self.evaluation_data['std_reward']),
                'success_rate': float(self.evaluation_data['success_rate']),
                'avg_length': float(self.evaluation_data['avg_length']),
                'comm_cost': float(self.evaluation_data['comm_cost'])
            },
            'config': self._namespace_to_dict(self.config)
        }

        summary_file = os.path.join(self.output_dir, "experiment_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def get_file_path(self, filename: str) -> str:
        """获取输出目录中的文件路径"""
        return os.path.join(self.output_dir, filename)

    def close(self):
        """关闭日志记录器"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class MultiExperimentLogger:
    """多实验日志记录器，用于汇总多个实验的结果"""

    def __init__(self, output_dir: str = "results"):
        """初始化多实验日志记录器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.experiments = []
        self.summary_data = []

        self.logger = logging.getLogger("MultiExperimentLogger")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def add_experiment(self, experiment_logger: ExperimentLogger):
        """添加实验记录器"""
        self.experiments.append(experiment_logger)

        # 提取摘要信息
        summary = {
            'algorithm': experiment_logger.algorithm_name,
            'environment': experiment_logger.env_name,
            'seed': experiment_logger.seed,
            'experiment_id': experiment_logger.experiment_id,
            'total_episodes': experiment_logger.total_episodes,
            'total_steps': experiment_logger.total_steps,
            'best_reward': experiment_logger.best_reward,
            'evaluation': experiment_logger.evaluation_data
        }

        self.summary_data.append(summary)

        self.logger.info(
            f"Added experiment: {experiment_logger.algorithm_name} "
            f"on {experiment_logger.env_name} (seed {experiment_logger.seed})"
        )

    def save_summary(self):
        """保存所有实验的汇总结果"""
        if not self.summary_data:
            self.logger.warning("No experiment data to save")
            return

        # 按算法和环境分组
        grouped_data = {}
        for summary in self.summary_data:
            key = f"{summary['algorithm']}_{summary['environment']}"
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(summary)

        # 计算统计信息
        results_summary = {}
        for key, experiments in grouped_data.items():
            algorithm, environment = key.split('_')

            # 计算平均值和标准差
            avg_rewards = [exp['evaluation']['avg_reward'] for exp in experiments]
            success_rates = [exp['evaluation']['success_rate'] for exp in experiments]
            comm_costs = [exp['evaluation']['comm_cost'] for exp in experiments]

            results_summary[key] = {
                'algorithm': algorithm,
                'environment': environment,
                'num_experiments': len(experiments),
                'avg_reward_mean': float(np.mean(avg_rewards)),
                'avg_reward_std': float(np.std(avg_rewards)),
                'success_rate_mean': float(np.mean(success_rates)),
                'success_rate_std': float(np.std(success_rates)),
                'comm_cost_mean': float(np.mean(comm_costs)),
                'comm_cost_std': float(np.std(comm_costs)),
                'seeds': [exp['seed'] for exp in experiments],
                'experiment_ids': [exp['experiment_id'] for exp in experiments]
            }

        # 保存汇总结果
        summary_file = os.path.join(
            self.output_dir,
            f"multi_experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results_summary': results_summary,
                'all_experiments': self.summary_data
            }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Multi-experiment summary saved to: {summary_file}")

        # 打印汇总结果
        self.logger.info("\n" + "="*60)
        self.logger.info("Multi-Experiment Summary")
        self.logger.info("="*60)

        for key, summary in results_summary.items():
            self.logger.info(
                f"\n{summary['algorithm']} on {summary['environment']}:"
                f"\n  Avg Reward: {summary['avg_reward_mean']:.2f} ± "
                f"{summary['avg_reward_std']:.2f}"
                f"\n  Success Rate: {summary['success_rate_mean']:.2%} ± "
                f"{summary['success_rate_std']:.2%}"
                f"\n  Comm Cost: {summary['comm_cost_mean']:.2f} ± "
                f"{summary['comm_cost_std']:.2f} KB/step"
                f"\n  Number of runs: {summary['num_experiments']}"
            )

        return results_summary


if __name__ == "__main__":
    # 测试日志记录器
    from experiments.utils.config import get_default_config

    config = get_default_config()
    logger = ExperimentLogger("TestAlgorithm", "TestEnv", 42, config)

    # 记录训练数据
    training_results = {
        'episode_rewards': [10, 20, 30, 40, 50],
        'episode_lengths': [100, 90, 80, 70, 60],
        'total_steps': 500,
        'total_episodes': 5,
        'best_reward': 50.0
    }
    logger.log_training(training_results)

    # 记录评估数据
    evaluation_results = {
        'avg_reward': 35.0,
        'std_reward': 5.0,
        'success_rate': 0.8,
        'avg_length': 75.0,
        'comm_cost': 2.5,
        'total_rewards': [30, 35, 40],
        'episode_lengths': [70, 75, 80]
    }
    logger.log_evaluation(evaluation_results)

    # 保存数据
    logger.save()
    logger.close()

    print("Logger test completed successfully!")