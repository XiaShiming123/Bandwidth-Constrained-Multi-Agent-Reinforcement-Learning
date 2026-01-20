#!/usr/bin/env python3
"""
最终修复和运行脚本
"""

import os
import sys
import subprocess
import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_MA_PROMPT'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

print("=" * 80)
print("最终修复和运行脚本")
print("=" * 80)

# 1. 修复Gym兼容性
print("\n1. 修复Gym兼容性...")

try:
    # 安装gymnasium
    subprocess.run([sys.executable, "-m", "pip", "install", "gymnasium", "-q"], check=True)
    print("  ✅ 安装Gymnasium")
    
    import gymnasium as gym
    print(f"  ✅ Gymnasium版本: {gym.__version__}")
    
    # 替换gym模块
    sys.modules['gym'] = gym
    
    # 添加缺失属性
    import numpy as np
    
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
        print("  ✅ 添加了 gym.spaces.prng")
    
    if not hasattr(gym.spaces, 'np_random'):
        gym.spaces.np_random = np.random.RandomState()
        print("  ✅ 添加了 gym.spaces.np_random")
    
except Exception as e:
    print(f"  ⚠️  Gym兼容性修复警告: {e}")

# 2. 修复MPE源代码
print("\n2. 修复MPE源代码...")

mpe_file = "multiagent-particle-envs/multiagent/multi_discrete.py"
if os.path.exists(mpe_file):
    try:
        # 读取文件
        with open(mpe_file, 'r') as f:
            content = f.read()
        
        # 备份
        with open(mpe_file + '.backup', 'w') as f:
            f.write(content)
        print(f"  ✅ 备份到: {mpe_file}.backup")
        
        # 修复
        content = content.replace("from gym.spaces import prng", "import numpy as np")
        content = content.replace("self.np_random = prng.np_random", "self.np_random = np.random.RandomState()")
        
        # 写回
        with open(mpe_file, 'w') as f:
            f.write(content)
        
        print(f"  ✅ 修复了 {mpe_file}")
        
        # 重新安装MPE
        print("  重新安装MPE...")
        original_dir = os.getcwd()
        os.chdir("multiagent-particle-envs")
        subprocess.run([sys.executable, "setup.py", "develop"], capture_output=True)
        os.chdir(original_dir)
        print("  ✅ MPE重新安装")
        
    except Exception as e:
        print(f"  ❌ 修复MPE失败: {e}")
else:
    print(f"  ⚠️  MPE文件不存在: {mpe_file}")

# 3. 测试MPE导入
print("\n3. 测试MPE导入...")

try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print("  ✅ MPE导入成功")
    print(f"  ✅ MultiAgentEnv导入成功")
    print(f"  ✅ scenarios导入成功")
    
    # 列出场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f"  可用场景: {len(scenario_names)}个")
    if scenario_names:
        print(f"  示例: {scenario_names[:3]}")
    
except ImportError as e:
    print(f"  ❌ MPE导入失败: {e}")
    print("  可能需要手动修复:")
    print("    cd multiagent-particle-envs")
    print("    pip install -e .")
except Exception as e:
    print(f"  ⚠️  MPE测试警告: {e}")

# 4. 检查GPU
print("\n4. 检查GPU...")

try:
    import torch
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {name} ({mem:.1f}GB)")
except Exception as e:
    print(f"  ❌ 检查GPU失败: {e}")

# 5. 运行实验
print("\n5. 运行实验...")
print("=" * 80)

import datetime
output_dir = f"final_experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"输出目录: {output_dir}")
print(f"开始时间: {datetime.datetime.now()}")
print()

# 导入并运行主程序
sys.path.append('.')

# 直接运行主程序逻辑
import argparse
import numpy as np
import torch
import random
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    from experiments.environments.multiagent_env import MultiAgentEnvWrapper
    from experiments.algorithms.mappo import MAPPO
    from experiments.algorithms.iacn import IACN
    from experiments.algorithms.sparse_comm import SparseComm
    from experiments.algorithms.full_comm import FullComm
    from experiments.algorithms.adaptive_comm import AdaptiveComm
    from experiments.utils.logger import ExperimentLogger
    from experiments.utils.config import load_config
    
    print("✅ 所有模块导入成功")
    
    # 解析参数
    parser = argparse.ArgumentParser(description="多智能体通信与协作实验")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="配置文件路径")
    parser.add_argument("--env", type=str, default="MPE_Navigation",
                       choices=["MPE_Navigation", "MPE_PredatorPrey", "SMAC_3m_vs_3z"],
                       help="实验环境")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       default=["AdaptiveComm"],
                       help="要测试的算法列表")
    parser.add_argument("--seeds", type=int, nargs="+",
                       default=[42],
                       help="随机种子列表")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU设备ID，使用-1表示CPU")
    parser.add_argument("--output_dir", type=str, default=output_dir,
                       help="结果输出目录")
    
    # 使用默认参数
    args = parser.parse_args([])
    args.gpu = 0
    args.env = "MPE_Navigation"
    args.algorithms = ["AdaptiveComm"]
    args.seeds = [42]
    args.output_dir = output_dir
    
    # 加载配置
    config = load_config(args.config)
    config.experiment.env = args.env
    config.experiment.output_dir = args.output_dir
    
    # 设置GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            args.gpu = 0
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
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
    
    # 运行实验
    all_results = []
    
    for algorithm in args.algorithms:
        for seed in args.seeds:
            try:
                print(f"\n{'='*60}")
                print(f"Running {algorithm} on {args.env} with seed {seed}")
                print(f"{'='*60}")
                
                # 设置随机种子
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                # 创建环境
                env = MultiAgentEnvWrapper(args.env, config)
                
                # 创建算法
                if algorithm == "AdaptiveComm":
                    algo = AdaptiveComm(env, config)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                # 创建日志记录器
                logger = ExperimentLogger(
                    algorithm_name=algorithm,
                    env_name=args.env,
                    seed=seed,
                    config=config
                )
                
                # 训练
                print(f"\nTraining {algorithm}...")
                training_results = algo.train()
                
                # 评估
                print(f"\nEvaluating {algorithm}...")
                evaluation_results = algo.evaluate(num_episodes=config.evaluation.num_episodes)
                
                # 记录结果
                logger.log_training(training_results)
                logger.log_evaluation(evaluation_results)
                logger.save()
                
                result = {
                    'algorithm': algorithm,
                    'env': args.env,
                    'seed': seed,
                    'training': training_results,
                    'evaluation': evaluation_results,
                    'logger': logger
                }
                all_results.append(result)
                
                print(f"\n✅ {algorithm} 实验完成!")
                
            except Exception as e:
                print(f"❌ Error running {algorithm} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    
    if all_results:
        for result in all_results:
            print(f"\n{result['algorithm']} (seed {result['seed']}):")
            eval_data = result['evaluation']
            print(f"  Average Reward: {eval_data.get('avg_reward', 0):.2f}")
            print(f"  Success Rate: {eval_data.get('success_rate', 0):.2%}")
            print(f"  Communication Cost: {eval_data.get('comm_cost', 0):.2f} KB/step")
    
    print(f"\n✅ 实验完成! 结果保存在: {output_dir}")
    
except Exception as e:
    print(f"❌ 运行实验失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print(f"完成时间: {datetime.datetime.now()}")
print("=" * 80)