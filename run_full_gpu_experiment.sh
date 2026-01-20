#!/bin/bash

# ============================================
# 在GPU服务器上运行完整论文实验
# ============================================

set -e

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  完整论文实验 - GPU模式${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# 1. 设置环境
cd /home/xxjss/code/xu_first
export SUPPRESS_MA_PROMPT=1
export PYTHONWARNINGS=ignore

echo -e "${GREEN}[1] 环境设置完成${NC}"
echo "   工作目录: $(pwd)"
echo

# 2. 检查GPU
echo -e "${GREEN}[2] 检查GPU...${NC}"
python3 -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {name} ({mem:.1f}GB)')
"
echo

# 3. 选择GPU ID
GPU_ID=0
if python3 -c "import torch; print(torch.cuda.device_count())" | grep -q "[1-9]"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}[3] 选择GPU ID (0-$((GPU_COUNT-1)))${NC}"
    read -p "输入GPU ID (默认: 0): " USER_GPU_ID
    if [[ ! -z "$USER_GPU_ID" ]]; then
        GPU_ID=$USER_GPU_ID
    fi
else
    echo -e "${GREEN}[3] 使用默认GPU ID: 0${NC}"
fi
echo "   使用GPU: $GPU_ID"
echo

# 4. 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="full_gpu_experiment_${TIMESTAMP}"
echo -e "${GREEN}[4] 创建输出目录${NC}"
echo "   输出目录: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo

# 5. 运行完整实验
echo -e "${GREEN}[5] 开始完整论文实验${NC}"
echo "   环境: MPE_Navigation"
echo "   算法: MAPPO, IACN, SparseComm, FullComm, AdaptiveComm"
echo "   种子: 42, 123, 456"
echo "   GPU: $GPU_ID"
echo "   输出: $OUTPUT_DIR"
echo

echo -e "${BLUE}实验开始时间: $(date)${NC}"
echo

# 运行实验
python3 experiments/main_experiment_fixed.py \
  --gpu "$GPU_ID" \
  --env MPE_Navigation \
  --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
  --seeds 42 123 456 \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "${OUTPUT_DIR}/experiment.log"

echo
echo -e "${BLUE}实验结束时间: $(date)${NC}"
echo

# 6. 检查结果
echo -e "${GREEN}[6] 检查实验结果${NC}"
if [ -f "${OUTPUT_DIR}/experiment_report.md" ]; then
    echo "=== 实验摘要 ==="
    head -100 "${OUTPUT_DIR}/experiment_report.md"
else
    # 查找结果文件
    echo "生成的文件:"
    find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.json" -o -name "*.md" | head -20
    
    # 检查日志最后部分
    echo "\n=== 实验日志最后部分 ==="
    tail -50 "${OUTPUT_DIR}/experiment.log"
fi
echo

# 7. GPU使用统计
echo -e "${GREEN}[7] GPU使用统计${NC}"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu \
           --format=csv -l 1 2>/dev/null | head -5
echo

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  实验完成!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "结果目录: $OUTPUT_DIR"
echo "实验日志: ${OUTPUT_DIR}/experiment.log"
echo "实时监控: tail -f ${OUTPUT_DIR}/experiment.log"
echo "GPU监控: watch -n 1 nvidia-smi"
echo
