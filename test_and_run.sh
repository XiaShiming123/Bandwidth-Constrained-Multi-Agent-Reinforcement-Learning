#!/bin/bash

# ============================================
# 测试并运行实验
# ============================================

cd /home/xxjss/code/xu_first

# 设置环境变量
export SUPPRESS_MA_PROMPT=1
export PYTHONWARNINGS=ignore

echo "========================================"
echo "测试Gym/MPE兼容性"
echo "========================================"
echo

# 1. 测试Gym兼容性
echo "1. 测试Gym兼容性..."
python3 -c "
import sys
import warnings
warnings.filterwarnings('ignore')

# 尝试导入gymnasium
try:
    import gymnasium as gym
    print(f'✅ Gymnasium版本: {gym.__version__}')
    sys.modules['gym'] = gym
    
    # 添加缺失属性
    import numpy as np
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
        print('✅ 添加了 gym.spaces.prng')
    
    if not hasattr(gym.spaces, 'np_random'):
        gym.spaces.np_random = np.random.RandomState()
        print('✅ 添加了 gym.spaces.np_random')
    
except ImportError:
    print('⚠️  Gymnasium未安装，使用gym')
    import gym
    print(f'✅ Gym版本: {gym.__version__}')
"
echo

# 2. 测试MPE导入
echo "2. 测试MPE导入..."
python3 -c "
import sys
import warnings
warnings.filterwarnings('ignore')

# 应用Gym兼容性修复
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
    import numpy as np
    class FakePRNG:
        def __init__(self):
            self.np_random = np.random.RandomState()
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = FakePRNG()
except:
    import gym

try:
    import multiagent
    print('✅ MPE导入成功')
    
    # 测试详细导入
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    print('✅ MultiAgentEnv导入成功')
    print('✅ scenarios导入成功')
    
    # 列出场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f'可用场景: {len(scenario_names)}个')
    if scenario_names:
        print(f'示例: {scenario_names[:3]}')
    
except ImportError as e:
    print(f'❌ MPE导入失败: {e}')
    print('可能需要修复MPE源代码:')
    print('  sed -i \"s/from gym.spaces import prng/import numpy as np/g\" multiagent-particle-envs/multiagent/multi_discrete.py')
    print('  sed -i \"s/self.np_random = prng.np_random/self.np_random = np.random.RandomState()/g\" multiagent-particle-envs/multiagent/multi_discrete.py')
except Exception as e:
    print(f'❌ MPE测试错误: {e}')
"
echo

# 3. 检查GPU
echo "3. 检查GPU..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {name} ({mem:.1f}GB)')
"
echo

# 4. 运行小规模实验
echo "4. 运行小规模实验..."
echo "========================================"
echo "开始实验"
echo "========================================"
echo

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="test_experiment_${TIMESTAMP}"

echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo

python3 experiments/main_experiment_final.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --output_dir "$OUTPUT_DIR"

echo
echo "========================================"
echo "实验完成"
echo "========================================"
echo

echo "完成时间: $(date)"
echo "结果目录: $OUTPUT_DIR"
echo

# 5. 检查结果
echo "5. 检查结果..."
if [ -f "${OUTPUT_DIR}/experiment_report.md" ]; then
    echo "=== 实验报告摘要 ==="
    head -30 "${OUTPUT_DIR}/experiment_report.md"
else
    echo "查找生成的文件:"
    find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.json" -o -name "*.md" 2>/dev/null | head -5
fi
echo

echo "========================================"
echo "测试完成"
echo "========================================"