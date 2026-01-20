"# 在CentOS 7服务器上运行论文实验

## 环境信息
- 操作系统: CentOS 7
- Python环境: Ramgmal (远程conda环境)
- 项目路径: `/home/xxjss/code/xu_first/`

## 步骤1: 检查环境

运行环境检查脚本:
```bash
cd /home/xxjss/code/xu_first
python check_env.py
```

## 步骤2: 安装缺失的依赖

如果检查显示缺少包，安装它们:
```bash
pip install numpy matplotlib pandas seaborn scipy gym pygame pyyaml tqdm scikit-learn
```

## 步骤3: 安装MPE环境

MPE (Multi-Agent Particle Environment) 是必需的:

```bash
cd /home/xxjss/code/xu_first

# 克隆MPE仓库
git clone https://github.com/openai/multiagent-particle-envs.git

# 如果github访问慢，可以使用镜像
# git clone https://gitee.com/mirrors/multiagent-particle-envs.git

# 安装MPE
cd multiagent-particle-envs
pip install -e .
cd ..
```

验证MPE安装:
```bash
python -c \"import multiagent; print('MPE安装成功')\"
```

## 步骤4: 运行实验

### 选项A: 快速测试 (验证框架)
```bash
# 测试算法框架
python test_framework_simple.py

# 运行示例实验
python run_example.py
```

### 选项B: 小规模实验 (验证GPU)
```bash
# 使用GPU (如果可用)
python experiments/main_experiment.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm MAPPO \
  --seeds 42 \
  --output_dir test_results

# 强制使用CPU
python experiments/main_experiment.py \
  --gpu -1 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --output_dir test_results_cpu
```

### 选项C: 完整论文实验
```bash
# 完整实验 (5个算法，3个随机种子)
python experiments/main_experiment.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
  --seeds 42 123 456 \
  --output_dir paper_results_full

# 追捕任务实验
python experiments/main_experiment.py \
  --gpu 0 \
  --env MPE_PredatorPrey \
  --algorithms AdaptiveComm IACN MAPPO \
  --seeds 42 123 \
  --output_dir paper_results_predatorprey
```

## 步骤5: 监控实验进度

### 查看日志
```bash
# 实时查看实验日志
tail -f paper_results_full/experiment.log

# 查看特定算法的训练进度
grep \"AdaptiveComm\" paper_results_full/experiment.log
```

### 监控GPU使用
```bash
# 查看GPU状态
nvidia-smi
nvidia-smi -l 5  # 每5秒刷新一次
```

### 监控内存使用
```bash
# 查看进程内存
top -u xxjss

# 查看Python进程
ps aux | grep python
```

## 步骤6: 分析结果

实验完成后，结果保存在指定的输出目录中:

### 生成的结果文件
- `performance_comparison.png` - 算法性能对比图
- `performance_vs_efficiency.png` - 性能-通信效率关系图
- `training_curves.png` - 训练曲线对比
- `experiment_report.md` - 详细实验报告
- `summary_*.json` - 实验结果汇总数据

### 查看结果
```bash
# 查看实验报告
cat paper_results_full/experiment_report.md

# 查看汇总数据
cat paper_results_full/summary_*.json
```

## 常见问题解决

### 1. CUDA不可用
```
错误: CUDA not available
解决:
- 检查nvidia驱动: nvidia-smi
- 检查CUDA版本: nvcc --version
- 重新安装PyTorch CUDA版本: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. MPE安装失败
```
错误: No module named 'multiagent'
解决:
- 确保在multiagent-particle-envs目录中运行安装
- 检查Python路径: python -c \"import sys; print(sys.path)\"
- 手动添加路径: export PYTHONPATH=/path/to/multiagent-particle-envs:$PYTHONPATH
```

### 3. 内存不足
```
错误: CUDA out of memory
解决:
- 减小batch_size (修改configs/default.yaml)
- 使用更小的网络
- 使用CPU模式: --gpu -1
```

### 4. 训练太慢
```
解决:
- 使用GPU加速
- 减小num_episodes进行测试
- 使用更简单的环境
```

## 性能优化建议

### GPU优化
1. **批量大小**: 根据GPU内存调整batch_size
2. **混合精度训练**: 使用torch.cuda.amp加速
3. **数据加载**: 使用DataLoader并行加载

### 内存优化
1. **梯度累积**: 模拟更大的batch_size
2. **检查点**: 定期保存模型，释放内存
3. **内存分析**: 使用torch.cuda.memory_summary()

### 实验管理
1. **使用screen/tmux**: 保持实验在后台运行
2. **定期备份**: 备份重要结果
3. **版本控制**: 记录实验配置

## 论文结果调整

基于实际运行结果，可能需要调整论文中的:

### 1. 实验参数
- 学习率、折扣因子等超参数
- 训练回合数
- 评估标准

### 2. 算法实现
- 通信成本计算方式
- 自适应机制参数
- 网络架构

### 3. 结果分析
- 性能对比数据
- 通信效率指标
- 收敛速度分析

## 联系支持

如果遇到问题:
1. 查看错误日志
2. 检查环境配置
3. 参考项目README
4. 联系系统管理员

祝实验顺利!"