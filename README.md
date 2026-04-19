"# 面向受限环境的集成自适应多智能体通信与协作框架 - 实验代码

## 项目概述

本项目实现了论文《面向受限环境的集成自适应多智能体通信与协作框架》的实验部分，包含完整的实验代码、算法实现和评估工具。

## 主要特性

- **完整算法实现**: 实现了5种多智能体通信算法
- **多种环境支持**: 支持MPE和SMAC环境
- **自适应通信**: 实现了拓扑、内容、频率三重自适应机制
- **全面评估**: 包含性能、通信效率、鲁棒性等多维度评估
- **可视化工具**: 自动生成实验图表和报告

## 算法列表

1. **AdaptiveComm** (本文方法): 集成自适应通信框架
   - 自适应拓扑选择
   - 自适应内容压缩
   - 自适应通信频率
   - 通信成本优化

2. **MAPPO**: 多智能体PPO算法（无通信基准）
   - 集中式critic
   - 分布式actor
   - 无智能体间通信

3. **IACN**: 独立演员通信网络
   - 自适应通信频率
   - 固定通信拓扑
   - 基于价值函数方差调整

4. **SparseComm**: 稀疏通信算法
   - 固定稀疏拓扑
   - 固定通信频率
   - 最近邻通信

5. **FullComm**: 完全通信算法（理论上限）
   - 全连接拓扑
   - 每步必通信
   - 无消息压缩

## 环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)

### 多智能体环境
- **MPE** (Multi-Agent Particle Environment): 用于导航和追捕任务
- **SMAC** (StarCraft II Multi-Agent Challenge): 用于战斗任务

## 快速开始

### 1. 安装依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# 运行安装脚本
python setup.py
```

### 2. 运行实验
```bash
# 基本实验（MPE导航环境）
python experiments/main_experiment.py

# 指定环境和算法
python experiments/main_experiment.py \
  --env MPE_Navigation \
  --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
  --seeds 42 123 456

# 追捕任务实验
python experiments/main_experiment.py \
  --env MPE_PredatorPrey \
  --algorithms AdaptiveComm IACN \
  --seeds 42 123

# SMAC战斗实验
python experiments/main_experiment.py \
  --env SMAC_3m_vs_3z \
  --algorithms AdaptiveComm MAPPO \
  --seeds 42 123
```

### 3. 查看结果
实验结果保存在 `results/` 目录，包含：
- 训练曲线图
- 性能对比图
- 通信效率分析
- 详细实验报告

## 项目结构

```
experiments/
├── algorithms/              # 算法实现
│   ├── adaptive_comm.py    # 自适应通信算法
│   ├── mappo.py           # MAPPO算法
│   ├── iacn.py            # IACN算法
│   ├── sparse_comm.py     # 稀疏通信算法
│   ├── full_comm.py       # 完全通信算法
│   └── base_algorithm.py  # 基础算法类
├── environments/          # 环境封装
│   └── multiagent_env.py  # 多智能体环境封装
├── utils/                # 工具类
│   ├── config.py         # 配置管理
│   ├── logger.py         # 日志记录
│   └── visualization.py  # 可视化工具
├── configs/              # 配置文件
│   └── default.yaml      # 默认配置
└── main_experiment.py    # 实验主程序

results/                  # 实验结果
models/                   # 训练好的模型
logs/                     # 日志文件
```

## 配置说明

### 主要配置参数

```yaml
# 实验设置
experiment:
  env: "MPE_Navigation"  # 环境名称
  output_dir: "results"  # 输出目录

# 训练参数
training:
  num_episodes: 1000     # 训练回合数
  learning_rate: 3e-4    # 学习率
  gamma: 0.99           # 折扣因子

# 通信约束
communication:
  bandwidth_limit: 10.0  # 带宽限制 (KB/步)
  latency: 1            # 延迟 (步)
  packet_loss: 0.1      # 丢包概率
```

### 算法特定参数

每个算法都有特定的配置参数，可以在配置文件中调整：

- **AdaptiveComm**: 通信维度、稀疏度阈值、拓扑更新频率
- **IACN**: 基础通信频率、拓扑类型
- **SparseComm**: 邻居数量、通信频率
- **MAPPO**: Critic网络维度

## 实验结果

### 预期性能

根据论文结果，各算法在MPE导航任务上的预期表现：

| 算法 | 平均奖励 | 成功率 | 通信成本 (KB/步) |
|------|----------|--------|------------------|
| AdaptiveComm | 85-90 | 85-90% | 4-5 |
| IACN | 70-75 | 70-75% | 8-9 |
| MAPPO | 65-70 | 65-70% | 0 |
| SparseComm | 60-65 | 60-65% | 15-16 |
| FullComm | 90-95 | 90-95% | 100-105 |

### 关键发现

1. **通信效率**: AdaptiveComm在通信成本降低60%的同时，性能接近FullComm
2. **自适应优势**: 三重自适应机制在动态环境中表现更佳
3. **可扩展性**: 算法在最多20个智能体的场景中仍保持良好性能
4. **鲁棒性**: 对通信干扰和智能体故障具有较好的容错能力

## 扩展和定制

### 添加新算法
1. 在 `experiments/algorithms/` 中创建新算法类
2. 继承 `BaseAlgorithm` 类
3. 实现 `select_action` 和 `train` 方法
4. 在配置文件中添加算法参数

### 添加新环境
1. 在 `experiments/environments/` 中创建环境封装
2. 实现标准接口（reset, step, get_state等）
3. 在 `MultiAgentEnvWrapper` 中添加环境初始化

### 自定义评估指标
1. 修改 `BaseAlgorithm.evaluate` 方法
2. 添加新的评估指标计算
3. 更新可视化工具以显示新指标

## 故障排除

### 常见问题

1. **环境安装失败**
   ```
   # MPE环境
   git clone https://github.com/openai/multiagent-particle-envs.git
   cd multiagent-particle-envs
   pip install -e .
   ```

2. **GPU内存不足**
   - 减小批量大小 (`training.batch_size`)
   - 减少智能体数量
   - 使用CPU模式 (`--gpu -1`)

3. **训练不稳定**
   - 调整学习率 (`training.learning_rate`)
   - 增加熵正则化系数 (`training.entropy_coef`)
   - 使用更多随机种子

### 调试模式
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 单次运行测试
python experiments/main_experiment.py \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --config experiments/configs/debug.yaml
```

## 引用

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@article{adaptive_multiagent_comm_2024,
  title={面向受限环境的集成自适应多智能体通信与协作框架},
  author={作者},
  journal={期刊名称},
  volume={卷},
  number={期},
  pages={页码},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: [your-email@example.com]
- GitHub Issues: [项目地址]/issues

## 更新日志

### v1.0.0 (2024-01-17)
- 初始版本发布
- 实现5种多智能体通信算法
- 支持MPE和SMAC环境
- 完整的实验和评估工具链"