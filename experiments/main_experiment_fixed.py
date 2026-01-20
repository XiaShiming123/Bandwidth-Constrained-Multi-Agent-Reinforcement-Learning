import gym_patch
#!/usr/bin/env python3
"""
修复版主实验程序 - 确保GPU参数正确解析
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from experiments.environments.multiagent_env import MultiAgentEnvWrapper
from experiments.algorithms.mappo import MAPPO
from experiments.algorithms.iacn import IACN
from experiments.algorithms.sparse_comm import SparseComm
from experiments.algorithms.full_comm import FullComm
from experiments.algorithms.adaptive_comm import AdaptiveComm
from experiments.utils.logger import ExperimentLogger
from experiments.utils.config import load_config


def set_seed(seed):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config, algorithm_name, env_name, seed):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running {algorithm_name} on {env_name} with seed {seed}")
    print(f"{'='*60}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 创建环境
    env = MultiAgentEnvWrapper(env_name, config)
    
    # 创建算法
    if algorithm_name == "MAPPO":
        algorithm = MAPPO(env, config)
    elif algorithm_name == "IACN":
        algorithm = IACN(env, config)
    elif algorithm_name == "SparseComm":
        algorithm = SparseComm(env, config)
    elif algorithm_name == "FullComm":
        algorithm = FullComm(env, config)
    elif algorithm_name == "AdaptiveComm":
        algorithm = AdaptiveComm(env, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # 创建日志记录器
    logger = ExperimentLogger(
        algorithm_name=algorithm_name,
        env_name=env_name,
        seed=seed,
        config=config
    )
    
    # 训练
    print(f"\nTraining {algorithm_name}...")
    training_results = algorithm.train()
    
    # 评估
    print(f"\nEvaluating {algorithm_name}...")
    evaluation_results = algorithm.evaluate(num_episodes=config.evaluation.num_episodes)
    
    # 记录结果
    logger.log_training(training_results)
    logger.log_evaluation(evaluation_results)
    logger.save()
    
    return {
        'algorithm': algorithm_name,
        'env': env_name,
        'seed': seed,
        'training': training_results,
        'evaluation': evaluation_results,
        'logger': logger
    }


def main():
    parser = argparse.ArgumentParser(description="多智能体通信与协作实验")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="配置文件路径")
    parser.add_argument("--env", type=str, default="MPE_Navigation",
                       choices=["MPE_Navigation", "MPE_PredatorPrey", "SMAC_3m_vs_3z"],
                       help="实验环境")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       default=["MAPPO", "IACN", "SparseComm", "FullComm", "AdaptiveComm"],
                       help="要测试的算法列表")
    parser.add_argument("--seeds", type=int, nargs="+",
                       default=[42, 123, 456, 789, 999],
                       help="随机种子列表")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU设备ID，使用-1表示CPU")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="结果输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config.experiment.env = args.env
    config.experiment.output_dir = args.output_dir
    
    # 设置GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        # 检查GPU ID是否有效
        if args.gpu >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu} not available, using GPU 0")
            args.gpu = 0
        
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    elif args.gpu >= 0 and not torch.cuda.is_available():
        print(f"Warning: GPU requested but CUDA not available, using CPU")
        device = "cpu"
        args.gpu = -1
    else:
        device = "cpu"
        print("Using CPU")
    
    config.device = device
    config.experiment.device = device
    
    print(f"Device: {config.device}")
    print(f"Environment: {args.env}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行所有实验
    all_results = []
    
    for algorithm in args.algorithms:
        for seed in args.seeds:
            try:
                result = run_experiment(config, algorithm, args.env, seed)
                all_results.append(result)
            except Exception as e:
                print(f"Error running {algorithm} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    
    # 计算平均性能
    summary = {}
    for algorithm in args.algorithms:
        algorithm_results = [r for r in all_results if r['algorithm'] == algorithm]
        if algorithm_results:
            avg_rewards = np.mean([r['evaluation']['avg_reward'] for r in algorithm_results])
            success_rates = np.mean([r['evaluation']['success_rate'] for r in algorithm_results])
            comm_costs = np.mean([r['evaluation']['comm_cost'] for r in algorithm_results])
            
            summary[algorithm] = {
                'avg_reward': avg_rewards,
                'success_rate': success_rates,
                'comm_cost': comm_costs,
                'num_runs': len(algorithm_results)
            }
            
            print(f"\n{algorithm}:")
            print(f"  Average Reward: {avg_rewards:.2f}")
            print(f"  Success Rate: {success_rates:.2%}")
            print(f"  Communication Cost: {comm_costs:.2f} KB/step")
            print(f"  Number of Runs: {len(algorithm_results)}")
    
    # 保存汇总结果
    summary_file = os.path.join(args.output_dir, f"summary_{args.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # 生成可视化
    from experiments.utils.visualization import create_summary_plots
    create_summary_plots(all_results, args.output_dir)
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()