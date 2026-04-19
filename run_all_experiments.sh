#!/bin/bash

# 面向受限环境的集成自适应多智能体通信与协作框架 - 运行所有实验脚本

# 设置GPU
GPU=4

# 设置随机种子
SEEDS="42 123 456 789 999"

# 设置算法列表
ALGORITHMS="MAPPO IACN SparseComm FullComm AdaptiveComm"

# 运行所有环境
for ENV in MPE_Navigation MPE_PredatorPrey SMAC_3m_vs_3z; do
    echo "=========================================="
    echo "Running experiments on $ENV..."
    echo "=========================================="

    for ALGORITHM in $ALGORITHMS; do
        echo "Running $ALGORITHM on $ENV..."
        python experiments/main_experiment.py --env $ENV --algorithms $ALGORITHM --seeds $SEEDS --gpu $GPU
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
