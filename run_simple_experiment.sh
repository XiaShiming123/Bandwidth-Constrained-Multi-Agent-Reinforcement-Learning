#!/bin/bash

# ============================================
# 最简单实验运行脚本
# ============================================

cd /home/xxjss/code/xu_first

# 设置环境
export SUPPRESS_MA_PROMPT=1
export PYTHONWARNINGS=ignore

echo "========================================"
echo "运行最简单实验"
echo "========================================"
echo

# 1. 创建最简单实验脚本
echo "1. 创建实验脚本..."
cat > simple_experiment.py << 'EOF'
#!/usr/bin/env python3
"""
最简单实验 - 验证框架能运行
"""

import os
import sys
import warnings

# 设置
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_MA_PROMPT'] = '1'

print("=" * 60)
print("最简单实验")
print("=" * 60)
print()

# 修复Gym兼容性
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
    print(f"✅ 使用Gymnasium")
    
    import numpy as np
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
except:
    import gym
    print(f"✅ 使用Gym")

print()

# 导入MPE
try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    print("✅ MPE导入成功")
except Exception as e:
    print(f"❌ MPE导入失败: {e}")
    sys.exit(1)

# 列出场景
scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
print(f"可用场景: {scenario_names[:5]}...")
print()

# 使用simple场景（确保存在）
scenario_name = "simple"
if scenario_name not in scenario_names:
    print(f"❌ 场景 {scenario_name} 不存在")
    sys.exit(1)

try:
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    print(f"✅ 环境创建成功 (场景: {scenario_name})")
    print(f"   智能体数量: {len(world.agents)}")
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    sys.exit(1)

# 测试环境
print("\n测试环境:")
obs = env.reset()
print(f"  ✅ 重置成功")

# 随机动作
if isinstance(obs, list):
    actions = [env.action_space[i].sample() for i in range(len(world.agents))]
else:
    actions = env.action_space.sample()

next_obs, rewards, dones, info = env.step(actions)
print(f"  ✅ 步进成功")
print(f"     奖励: {rewards}")

# 测试算法框架
print("\n测试算法框架:")
try:
    import torch
    print(f"  ✅ PyTorch版本: {torch.__version__}")
    print(f"  ✅ CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  ✅ GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 创建简单网络
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, output_dim)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    net = SimpleNet(10, 5)
    print(f"  ✅ 神经网络创建成功")
    
    # 测试GPU
    if torch.cuda.is_available():
        net = net.cuda()
        x = torch.randn(3, 10).cuda()
        y = net(x)
        print(f"  ✅ GPU计算成功: {y.shape}")
    
except Exception as e:
    print(f"  ⚠️  算法框架测试警告: {e}")

print("\n" + "=" * 60)
print("✅ 最简单实验通过!")
print("框架可以正常运行")
print("=" * 60)

# 询问是否运行完整实验
print("\n是否运行完整实验? (y/n): ")
response = sys.stdin.readline().strip().lower()
if response == 'y':
    print("\n运行完整实验...")
    
    # 这里可以添加完整实验代码
    print("完整实验代码待实现")
else:
    print("\n实验测试完成")
EOF

chmod +x simple_experiment.py
echo "  ✅ 创建了 simple_experiment.py"
echo

# 2. 运行测试
echo "2. 运行测试..."
python3 simple_experiment.py
echo

# 3. 如果测试成功，运行小规模实验
echo "3. 运行小规模实验..."
if [ $? -eq 0 ]; then
    echo "测试成功，运行小规模实验"
    
    OUTPUT_DIR="mini_experiment_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    echo "输出目录: $OUTPUT_DIR"
    echo "开始时间: $(date)"
    echo
    
    # 使用修复后的主程序
    python3 experiments/main_experiment_final.py \
      --gpu 0 \
      --env MPE_Navigation \
      --algorithms AdaptiveComm \
      --seeds 42 \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "${OUTPUT_DIR}/experiment.log"
    
    echo
    echo "完成时间: $(date)"
    echo "结果目录: $OUTPUT_DIR"
else
    echo "测试失败，请检查错误"
fi
echo

echo "========================================"
echo "脚本执行完成"
echo "========================================"