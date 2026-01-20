#!/bin/bash

# ============================================
# 一键修复脚本
# 解决Gym/MPE兼容性问题
# ============================================

set -e

cd /home/xxjss/code/xu_first

echo "========================================"
echo "一键修复Gym/MPE兼容性问题"
echo "========================================"
echo

# 1. 安装Gymnasium
echo "1. 安装Gymnasium..."
pip install gymnasium -q
python3 -c "import gymnasium; print(f'✅ Gymnasium {gymnasium.__version__} 已安装')"
echo

# 2. 修复MPE源代码
echo "2. 修复MPE源代码..."

MPE_FILE="multiagent-particle-envs/multiagent/multi_discrete.py"
if [ -f "$MPE_FILE" ]; then
    echo "  修复 $MPE_FILE"
    
    # 备份
    cp "$MPE_FILE" "${MPE_FILE}.backup"
    
    # 读取内容
    content=$(cat "$MPE_FILE")
    
    # 修复导入
    content=${content//"from gym.spaces import prng"/"import numpy as np"}
    
    # 修复prng使用
    content=${content//"self.np_random = prng.np_random"/"self.np_random = np.random.RandomState()"}
    
    # 写回
    echo "$content" > "$MPE_FILE"
    
    echo "  ✅ 修复完成"
else
    echo "  ⚠️  文件不存在: $MPE_FILE"
    echo "  尝试查找文件..."
    find . -name "multi_discrete.py" -type f 2>/dev/null | head -3
fi
echo

# 3. 重新安装MPE
echo "3. 重新安装MPE..."
if [ -d "multiagent-particle-envs" ]; then
    cd multiagent-particle-envs
    pip install -e . -q
    cd ..
    echo "  ✅ MPE重新安装完成"
else
    echo "  ⚠️  MPE目录不存在"
fi
echo

# 4. 创建并应用猴子补丁
echo "4. 创建猴子补丁..."
cat > gym_patch.py << 'EOF'
"""
Gym猴子补丁 - 解决MPE兼容性问题
在程序开始时导入此模块
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# 使用gymnasium替换gym
import gymnasium as gym
sys.modules['gym'] = gym

# 为旧版MPE添加缺失的prng属性
import numpy as np

class FakePRNG:
    def __init__(self):
        self.np_random = np.random.RandomState()

# 确保gym.spaces有prng属性
if not hasattr(gym.spaces, 'prng'):
    gym.spaces.prng = FakePRNG()

# 确保有np_random属性
if not hasattr(gym.spaces, 'np_random'):
    gym.spaces.np_random = np.random.RandomState()

print("✅ Gym猴子补丁已应用: 使用Gymnasium替代Gym")
EOF

echo "  ✅ 创建了 gym_patch.py"
echo

# 5. 修改主程序以应用补丁
echo "5. 修改主程序..."
MAIN_FILE="experiments/main_experiment_fixed.py"
if [ -f "$MAIN_FILE" ]; then
    # 在文件开头添加导入
    sed -i '1iimport gym_patch' "$MAIN_FILE"
    echo "  ✅ 修改了 $MAIN_FILE"
else
    echo "  ⚠️  文件不存在: $MAIN_FILE"
fi
echo

# 6. 测试修复
echo "6. 测试修复..."
python3 -c "
import gym_patch

try:
    import multiagent
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    
    print('✅ MPE导入成功!')
    
    # 列出场景
    scenario_names = [name for name in dir(scenarios) if not name.startswith('_')]
    print(f'可用场景: {len(scenario_names)}个')
    
    if scenario_names:
        print(f'示例: {scenario_names[:3]}')
    
except Exception as e:
    print(f'❌ MPE导入失败: {e}')
    import traceback
    traceback.print_exc()
"
echo

# 7. 运行测试
echo "7. 运行框架测试..."
python3 test_framework_simple.py 2>&1 | tail -20
echo

# 8. 运行实验的命令
echo "8. 运行实验的命令:"
echo "========================================"
echo "# 小规模测试"
echo "python experiments/main_experiment_fixed.py --gpu 0 --env MPE_Navigation --algorithms AdaptiveComm --seeds 42"
echo

echo "# 完整论文实验"
echo "python experiments/main_experiment_fixed.py --gpu 0 --env MPE_Navigation --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm --seeds 42 123 456"
echo

echo "========================================"
echo "修复完成!"
echo "========================================"