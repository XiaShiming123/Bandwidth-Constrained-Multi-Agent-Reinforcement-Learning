#!/bin/bash

# ============================================
# 最终实验运行脚本
# ============================================

cd /home/xxjss/code/xu_first

# 设置环境
export SUPPRESS_MA_PROMPT=1
export PYTHONWARNINGS=ignore

echo "========================================"
echo "最终实验运行"
echo "========================================"
echo

# 1. 测试MPE修复
echo "1. 测试MPE修复..."
python3 test_mpe_fix.py 2>&1 | tail -50
echo

# 检查测试结果
if [ $? -ne 0 ]; then
    echo "❌ MPE测试失败，请先修复问题"
    exit 1
fi

echo "✅ MPE测试通过"
echo

# 2. 创建最终实验脚本
echo "2. 创建最终实验脚本..."
cat > run_final_experiment.py << 'EOF'
#!/usr/bin/env python3
"""
最终实验脚本
"""

import os
import sys
import warnings
import argparse
import numpy as np
import torch
import random
from datetime import datetime

# 设置
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_MA_PROMPT'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

print("=" * 80)
print("面向受限环境的集成自适应多智能体通信与协作框架")
print("最终实验运行")
print("=" * 80)
print()

# 修复Gym兼容性
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
    print(f"✅ 使用Gymnasium版本: {gym.__version__}")
    
    import numpy as np
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
        print("✅ 添加了gym.spaces.prng属性")
except:
    import gym
    print(f"✅ 使用Gym版本: {gym.__version__}")

print()

# 添加项目路径
sys.path.append('.')

# 导入模块
try:
    from experiments.environments.multiagent_env import MultiAgentEnvWrapper
    from experiments.algorithms.adaptive_comm import AdaptiveComm
    from experiments.utils.logger import ExperimentLogger
    from experiments.utils.config import load_config
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

print()

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--env", type=str, default="MPE_Navigation")
parser.add_argument("--algorithm", type=str, default="AdaptiveComm")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="final_results")
parser.add_argument("--episodes", type=int, default=100)  # 测试用，减少回合数

args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 设置GPU
if args.gpu >= 0 and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        args.gpu = 0
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    gpu_name = torch.cuda.get_device_name(args.gpu)
    print(f"✅ 使用GPU: {args.gpu} ({gpu_name})")
else:
    device = "cpu"
    print("⚠️  使用CPU模式")

print()

# 加载配置
try:
    config = load_config("experiments/configs/default.yaml")
except:
    print("⚠️  使用默认配置")
    
    class SimpleConfig:
        class experiment:
            env = args.env
            output_dir = args.output_dir
            device = device
        class environment:
            max_steps = 100
            num_agents = 3
        class communication:
            bandwidth_limit = 10.0
            latency = 1
            packet_loss = 0.1
        class training:
            num_episodes = args.episodes
            max_steps = 100
            buffer_size = 1024
            batch_size = 32
            gamma = 0.99
            gae_lambda = 0.95
            ppo_clip = 0.2
            value_coef = 0.5
            entropy_coef = 0.01
            max_grad_norm = 0.5
            learning_rate = 3e-4
            log_interval = 5
        class evaluation:
            num_episodes = 5
            success_threshold = 50.0
        class adaptive_comm:
            comm_dim = 32
            comm_cost_coef = 0.1
            sparsity_threshold = 0.3
            topology_update_freq = 10
            base_neighbors = 2
    
    config = SimpleConfig()

config.device = device
config.experiment.device = device

print(f"实验配置:")
print(f"  环境: {args.env}")
print(f"  算法: {args.algorithm}")
print(f"  种子: {args.seed}")
print(f"  训练回合: {config.training.num_episodes}")
print(f"  设备: {device}")
print(f"  输出目录: {args.output_dir}")
print()

# 运行实验
try:
    print("=" * 60)
    print(f"开始实验")
    print("=" * 60)
    print()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建环境
    print("创建环境...")
    env = MultiAgentEnvWrapper(args.env, config)
    print(f"✅ 环境创建成功")
    print(f"   智能体数量: {env.num_agents}")
    print(f"   观测维度: {env.obs_dim}")
    print(f"   动作维度: {env.action_dim}")
    print()
    
    # 创建算法
    print("创建算法...")
    algorithm = AdaptiveComm(env, config)
    print(f"✅ 算法创建成功")
    print()
    
    # 创建日志记录器
    logger = ExperimentLogger(
        algorithm_name=args.algorithm,
        env_name=args.env,
        seed=args.seed,
        config=config
    )
    
    # 训练
    print("开始训练...")
    print("-" * 40)
    
    training_results = algorithm.train()
    
    print("-" * 40)
    print(f"✅ 训练完成")
    print(f"   总步数: {training_results.get('total_steps', 0)}")
    print(f"   总回合: {training_results.get('total_episodes', 0)}")
    print(f"   平均奖励: {training_results.get('avg_reward', 0):.2f}")
    print()
    
    # 评估
    print("开始评估...")
    evaluation_results = algorithm.evaluate(num_episodes=config.evaluation.num_episodes)
    
    print(f"✅ 评估完成")
    print(f"   平均奖励: {evaluation_results.get('avg_reward', 0):.2f}")
    print(f"   成功率: {evaluation_results.get('success_rate', 0):.2%}")
    print(f"   通信成本: {evaluation_results.get('comm_cost', 0):.2f} KB/步")
    print()
    
    # 记录结果
    logger.log_training(training_results)
    logger.log_evaluation(evaluation_results)
    logger.save()
    
    # 保存简单报告
    report_file = os.path.join(args.output_dir, "simple_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"实验报告\n")
        f.write(f"========\n\n")
        f.write(f"算法: {args.algorithm}\n")
        f.write(f"环境: {args.env}\n")
        f.write(f"种子: {args.seed}\n")
        f.write(f"设备: {device}\n")
        f.write(f"训练回合: {config.training.num_episodes}\n\n")
        f.write(f"训练结果:\n")
        f.write(f"  总步数: {training_results.get('total_steps', 0)}\n")
        f.write(f"  总回合: {training_results.get('total_episodes', 0)}\n")
        f.write(f"  平均奖励: {training_results.get('avg_reward', 0):.2f}\n\n")
        f.write(f"评估结果:\n")
        f.write(f"  平均奖励: {evaluation_results.get('avg_reward', 0):.2f}\n")
        f.write(f"  成功率: {evaluation_results.get('success_rate', 0):.2%}\n")
        f.write(f"  通信成本: {evaluation_results.get('comm_cost', 0):.2f} KB/步\n")
    
    print(f"✅ 实验完成!")
    print(f"结果保存在: {args.output_dir}")
    print(f"报告文件: {report_file}")
    
except Exception as e:
    print(f"\n❌ 实验失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("实验结束")
print(f"时间: {datetime.now()}")
print("=" * 80)
EOF

chmod +x run_final_experiment.py
echo "  ✅ 创建了 run_final_experiment.py"
echo

# 3. 运行实验
echo "3. 运行实验..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiment_${TIMESTAMP}"

echo "   输出目录: $OUTPUT_DIR"
echo "   开始时间: $(date)"
echo

python3 run_final_experiment.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithm AdaptiveComm \
  --seed 42 \
  --episodes 50 \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "${OUTPUT_DIR}/experiment.log"

echo
echo "完成时间: $(date)"
echo "结果目录: $OUTPUT_DIR"
echo

# 4. 检查结果
echo "4. 检查结果..."
if [ -f "${OUTPUT_DIR}/simple_report.txt" ]; then
    echo "=== 实验报告 ==="
    cat "${OUTPUT_DIR}/simple_report.txt"
else
    echo "生成的文件:"
    find "$OUTPUT_DIR" -type f 2>/dev/null | head -10
    
    echo "\n=== 实验日志最后部分 ==="
    tail -30 "${OUTPUT_DIR}/experiment.log"
fi
echo

echo "========================================"
echo "脚本执行完成"
echo "========================================"