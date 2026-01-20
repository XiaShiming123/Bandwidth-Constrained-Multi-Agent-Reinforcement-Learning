#!/bin/bash

# 立即运行GPU实验
cd /home/xxjss/code/xu_first

# 设置环境变量抑制警告
export SUPPRESS_MA_PROMPT=1
export PYTHONWARNINGS=ignore

# 检查GPU
python3 -c "
import torch
print('='*60)
print('GPU状态检查')
print('='*60)
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {name} ({mem:.1f}GB)')
print('='*60)
"

# 创建输出目录
OUTPUT_DIR="gpu_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "\n开始运行完整论文实验..."
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo

# 运行修复版主程序
python3 experiments/main_experiment_fixed.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
  --seeds 42 123 456 \
  --output_dir "$OUTPUT_DIR"

echo "\n实验完成时间: $(date)"
echo "结果保存在: $OUTPUT_DIR"

# 显示结果摘要
if [ -f "$OUTPUT_DIR/experiment_report.md" ]; then
    echo "\n=== 实验报告摘要 ==="
    head -50 "$OUTPUT_DIR/experiment_report.md"
fi

# 显示生成的文件
echo "\n=== 生成的文件 ==="
find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.json" -o -name "*.md" | sort