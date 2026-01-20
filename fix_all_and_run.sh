#!/bin/bash

# ============================================
# 完整修复和运行脚本
# 解决Gym/MPE兼容性问题并运行实验
# ============================================

set -e

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  完整修复和运行脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# 1. 设置环境
cd /home/xxjss/code/xu_first
export SUPPRESS_MA_PROMPT=1
export PYTHONWARNINGS=ignore

echo -e "${GREEN}[1] 环境设置完成${NC}"
echo "   工作目录: $(pwd)"
echo

# 2. 检查Gym版本
echo -e "${GREEN}[2] 检查Gym/Gymnasium版本...${NC}"
python3 -c "
import sys
try:
    import gym
    print(f'Gym版本: {gym.__version__}')
except ImportError:
    print('Gym: 未安装')
except Exception as e:
    print(f'Gym导入错误: {e}')

try:
    import gymnasium
    print(f'Gymnasium版本: {gymnasium.__version__}')
except ImportError:
    print('Gymnasium: 未安装')
except Exception as e:
    print(f'Gymnasium导入错误: {e}')
"
echo

# 3. 安装Gymnasium
echo -e "${GREEN}[3] 安装Gymnasium...${NC}"
pip install gymnasium[all] -q
python3 -c "import gymnasium; print(f'✅ Gymnasium {gymnasium.__version__} 已安装')"
echo

# 4. 创建Gym兼容层
echo -e "${GREEN}[4] 创建Gym兼容层...${NC}"
cat > gym_compat.py << 'EOF'
#!/usr/bin/env python3
"""
Gym兼容层 - 将Gym调用重定向到Gymnasium
解决MPE与新版Gym的兼容性问题
"""

import sys
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 尝试导入gymnasium
try:
    import gymnasium as gym
    print(f"✅ 使用Gymnasium版本: {gym.__version__}")
except ImportError:
    print("❌ Gymnasium未安装，尝试导入gym")
    import gym
    print(f"⚠️  使用旧版Gym: {gym.__version__}")

# 将gym模块替换为gymnasium
sys.modules['gym'] = gym

# 为旧版MPE添加缺失的属性
if hasattr(gym, 'spaces'):
    import numpy as np
    
    # 创建伪prng模块
    class PRNGWrapper:
        def __init__(self):
            self.np_random = np.random.RandomState()
    
    # 添加到gym.spaces
    if not hasattr(gym.spaces, 'prng'):
        gym.spaces.prng = PRNGWrapper()
    
    # 确保有np_random属性
    if not hasattr(gym.spaces, 'np_random'):
        gym.spaces.np_random = np.random.RandomState()

print("✅ Gym兼容层已激活")
EOF

chmod +x gym_compat.py
python3 gym_compat.py
echo

# 5. 修复MPE源代码
echo -e "${GREEN}[5] 修复MPE源代码...${NC}"

MPE_DIR="multiagent-particle-envs"
if [ ! -d "$MPE_DIR" ]; then
    echo -e "${YELLOW}MPE目录不存在，跳过修复${NC}"
else
    # 修复multi_discrete.py
    MPE_MULTI_DISCRETE="$MPE_DIR/multiagent/multi_discrete.py"
    if [ -f "$MPE_MULTI_DISCRETE" ]; then
        echo "  修复 $MPE_MULTI_DISCRETE"
        
        # 备份
        cp "$MPE_MULTI_DISCRETE" "${MPE_MULTI_DISCRETE}.backup"
        
        # 使用sed修复
        sed -i "s/from gym.spaces import prng/import numpy as np/g" "$MPE_MULTI_DISCRETE"
        sed -i "s/self.np_random = prng.np_random/self.np_random = np.random.RandomState()/g" "$MPE_MULTI_DISCRETE"
        
        echo "  ✅ multi_discrete.py 已修复"
    else
        echo -e "${YELLOW}  multi_discrete.py 不存在${NC}"
    fi
    
    # 重新安装MPE
    echo "  重新安装MPE..."
    cd "$MPE_DIR"
    pip install -e . -q
    cd ".."
    echo "  ✅ MPE重新安装完成"
fi
echo

# 6. 测试MPE导入
echo -e "${GREEN}[6] 测试MPE导入...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')

# 导入兼容层
import gym_compat

# 测试导入MPE
try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print('✅ MPE导入成功')
    
    # 列出场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f'可用场景: {len(scenario_names)}个')
    if scenario_names:
        print(f'示例: {scenario_names[:3]}')
    
except ImportError as e:
    print(f'❌ MPE导入失败: {e}')
    print('可能需要重新安装MPE: cd multiagent-particle-envs && pip install -e .')
except Exception as e:
    print(f'❌ MPE测试错误: {e}')
"
echo

# 7. 检查GPU
echo -e "${GREEN}[7] 检查GPU...${NC}"
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

# 8. 运行实验
echo -e "${GREEN}[8] 运行实验...${NC}"

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="fixed_experiment_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "   输出目录: $OUTPUT_DIR"
echo "   开始时间: $(date)"
echo

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  开始运行完整论文实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# 运行修复版主程序
python3 experiments/main_experiment_fixed.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
  --seeds 42 123 456 \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "${OUTPUT_DIR}/experiment.log"

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  实验完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo

echo "完成时间: $(date)"
echo "结果目录: $OUTPUT_DIR"
echo "实验日志: ${OUTPUT_DIR}/experiment.log"
echo

# 9. 检查结果
echo -e "${GREEN}[9] 检查实验结果...${NC}"
if [ -f "${OUTPUT_DIR}/experiment_report.md" ]; then
    echo "=== 实验报告摘要 ==="
    head -50 "${OUTPUT_DIR}/experiment_report.md"
else
    echo "查找生成的文件:"
    find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.json" -o -name "*.md" | head -10
fi
echo

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  脚本执行完成${NC}"
echo -e "${BLUE}========================================${NC}"