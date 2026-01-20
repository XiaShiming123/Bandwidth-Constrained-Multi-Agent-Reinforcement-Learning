# GPU/CUDA问题解决方案指南

## 问题诊断

根据之前的检查，远程服务器上PyTorch CUDA不可用。需要诊断具体原因：

### 1. 运行详细检查
```bash
cd /home/xxjss/code/xu_first
python check_cuda_detailed.py
```

### 2. 关键检查点
- 是否有NVIDIA GPU硬件
- NVIDIA驱动是否安装
- CUDA工具包是否安装
- PyTorch CUDA版本是否匹配

## 解决方案

### 方案A: 修复现有环境（需要管理员权限）

如果服务器有GPU但配置不正确：

#### 1. 检查GPU硬件
```bash
# 检查是否有NVIDIA GPU
lspci | grep -i nvidia

# 如果没有输出，说明没有NVIDIA GPU
# 如果有输出，记录GPU型号
```

#### 2. 检查驱动
```bash
# 检查NVIDIA驱动
nvidia-smi

# 如果命令不存在，需要安装驱动
# CentOS 7安装驱动:
sudo yum install epel-release
sudo yum install dkms
# 从NVIDIA官网下载对应驱动
```

#### 3. 检查CUDA
```bash
# 检查CUDA
nvcc --version

# 如果不存在，安装CUDA
# CUDA 11.8安装:
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
sudo yum clean all
sudo yum install -y cuda
```

#### 4. 重新安装PyTorch CUDA版本
```bash
# 卸载现有PyTorch
pip uninstall torch torchvision torchaudio -y

# 根据CUDA版本安装
# CUDA 11.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 方案B: 使用CPU模式（临时解决方案）

如果无法修复GPU，可以使用CPU模式运行实验：

#### 1. 修改实验配置
编辑 `experiments/configs/default.yaml`:
```yaml
experiment:
  device: "cpu"  # 改为cpu

training:
  num_episodes: 500  # 减少训练回合数（CPU较慢）
  batch_size: 32     # 减小批量大小
```

#### 2. 运行CPU实验
```bash
cd /home/xxjss/code/xu_first
export SUPPRESS_MA_PROMPT=1

# 小规模CPU实验
python experiments/main_experiment.py \
  --gpu -1 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --output_dir cpu_results
```

#### 3. 优化CPU性能
- 使用更小的网络架构
- 减少智能体数量
- 使用更简单的环境

### 方案C: 申请GPU资源

如果服务器没有GPU或无法修复：

#### 1. 校内GPU资源
- 联系实验室管理员
- 申请计算集群账号
- 使用学校GPU服务器

#### 2. 云GPU服务
- **Google Colab**: 免费GPU（有限制）
- **AWS EC2**: g4dn.xlarge（约$0.526/小时）
- **Azure NC系列**: NC6（约$0.90/小时）
- **阿里云**: gn6i（约¥4.5/小时）

#### 3. Colab运行指南
```python
# 在Google Colab中运行
!git clone https://github.com/your-repo/xu_first.git
%cd xu_first
!pip install -r requirements.txt
!git clone https://github.com/openai/multiagent-particle-envs.git
%cd multiagent-particle-envs
!pip install -e .
%cd ..

# 运行实验
!python experiments/main_experiment.py --gpu 0 --env MPE_Navigation --algorithms AdaptiveComm
```

## 实验调整建议

### GPU vs CPU性能对比

| 配置 | 训练速度 | 内存使用 | 适合场景 |
|------|----------|----------|----------|
| GPU (V100) | 10-100x | 高 | 完整实验，大批量 |
| GPU (T4) | 5-20x | 中 | 标准实验 |
| CPU (多核) | 1x | 低 | 测试、调试 |
| CPU (单核) | 0.1x | 很低 | 极小规模测试 |

### 根据资源调整实验

#### 充足GPU资源
```bash
# 完整论文实验
python experiments/main_experiment.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
  --seeds 42 123 456 789 999 \
  --output_dir full_results
```

#### 有限GPU资源
```bash
# 重点算法对比
python experiments/main_experiment.py \
  --gpu 0 \
  --env MPE_Navigation \
  --algorithms MAPPO IACN AdaptiveComm \
  --seeds 42 123 456 \
  --output_dir core_results
```

#### 仅CPU资源
```bash
# 简化实验
python experiments/main_experiment.py \
  --gpu -1 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --output_dir cpu_test \
  --config experiments/configs/cpu_optimized.yaml
```

## 创建CPU优化配置

创建 `experiments/configs/cpu_optimized.yaml`:

```yaml
experiment:
  env: "MPE_Navigation"
  output_dir: "results_cpu"
  device: "cpu"

environment:
  max_steps: 50  # 减少最大步数
  num_agents: 3  # 减少智能体数量

training:
  num_episodes: 500     # 减少训练回合数
  max_steps: 50        # 减少每回合步数
  batch_size: 32       # 减小批量大小
  buffer_size: 1024    # 减小缓冲区
  learning_rate: 0.0003

adaptive_comm:
  comm_dim: 16         # 减小通信维度
  base_neighbors: 1    # 减少基础邻居数
```

## 紧急解决方案

### 立即开始实验
如果急需开始实验，使用最小配置：

```bash
cd /home/xxjss/code/xu_first

# 创建最小化配置
echo '
experiment:
  device: "cpu"
training:
  num_episodes: 100
  batch_size: 16
' > min_config.yaml

# 运行最小实验
export SUPPRESS_MA_PROMPT=1
python experiments/main_experiment.py \
  --gpu -1 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --config min_config.yaml \
  --output_dir quick_test
```

### 验证算法正确性
先验证算法在CPU上能运行，再解决GPU问题：

```bash
# 验证框架
python test_framework_simple.py

# 运行示例
python run_example.py

# 小规模训练测试
python experiments/main_experiment.py \
  --gpu -1 \
  --env MPE_Navigation \
  --algorithms AdaptiveComm \
  --seeds 42 \
  --training.num_episodes 10 \
  --training.max_steps 20 \
  --output_dir verify_test
```

## 联系支持

### 校内支持
1. **计算中心**: 申请GPU资源
2. **实验室管理员**: 检查服务器配置
3. **导师**: 申请研究经费购买云GPU

### 外部资源
1. **Google Colab**: https://colab.research.google.com/
2. **AWS Educate**: 学生免费额度
3. **Microsoft Azure for Students**: 免费额度

### 社区帮助
1. **PyTorch论坛**: https://discuss.pytorch.org/
2. **Stack Overflow**: CUDA相关问题
3. **GitHub Issues**: 项目相关问题

## 下一步行动

### 短期（今天）
1. 运行 `check_cuda_detailed.py` 诊断问题
2. 使用CPU模式运行验证实验
3. 联系管理员询问GPU可用性

### 中期（本周）
1. 修复GPU环境或申请GPU资源
2. 运行完整CPU实验收集初步数据
3. 准备云GPU方案

### 长期（本月）
1. 在GPU上运行完整实验
2. 收集足够数据验证论文
3. 优化算法性能

记住：**先让实验跑起来，再优化性能**。CPU模式虽然慢，但可以验证算法正确性和实验流程。