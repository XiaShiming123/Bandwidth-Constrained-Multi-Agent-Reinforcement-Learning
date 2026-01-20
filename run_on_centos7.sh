#!/bin/bash

# ============================================
# 在CentOS 7上运行论文实验的脚本
# ============================================

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 $1 未找到"
        return 1
    fi
    return 0
}

# 主函数
main() {
    echo "=" 80
    echo "面向受限环境的集成自适应多智能体通信与协作框架"
    echo "CentOS 7实验运行脚本"
    echo "=" 80
    echo
    
    # 检查基本命令
    log_info "检查基本命令..."
    check_command python
    check_command pip
    check_command git
    
    # 切换到项目目录
    PROJECT_DIR="/home/xxjss/code/xu_first"
    if [ -d "$PROJECT_DIR" ]; then
        cd "$PROJECT_DIR"
        log_info "切换到项目目录: $(pwd)"
    else
        log_error "项目目录不存在: $PROJECT_DIR"
        exit 1
    fi
    
    # 步骤1: 检查环境
    log_info "步骤1: 检查Python环境..."
    python check_env.py
    echo
    
    # 询问是否继续
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "用户取消"
        exit 0
    fi
    
    # 步骤2: 检查MPE
    log_info "步骤2: 检查MPE环境..."
    if python -c "import multiagent" 2>/dev/null; then
        log_info "MPE已安装"
    else
        log_warn "MPE未安装"
        
        read -p "是否安装MPE? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_mpe
        else
            log_warn "跳过MPE安装，实验可能失败"
        fi
    fi
    
    # 步骤3: 选择实验模式
    echo
    log_info "步骤3: 选择实验模式"
    echo "1. 快速测试 (验证框架)"
    echo "2. 小规模实验 (验证GPU)"
    echo "3. 完整论文实验"
    echo "4. 自定义实验"
    echo
    
    read -p "请选择模式 (1-4): " mode
    echo
    
    case $mode in
        1)
            run_quick_test
            ;;
        2)
            run_small_experiment
            ;;
        3)
            run_full_experiment
            ;;
        4)
            run_custom_experiment
            ;;
        *)
            log_error "无效选择"
            exit 1
            ;;
    esac
    
    log_info "实验完成!"
}

# 安装MPE
install_mpe() {
    log_info "安装MPE环境..."
    
    # 克隆仓库
    if [ ! -d "multiagent-particle-envs" ]; then
        log_info "克隆MPE仓库..."
        git clone https://github.com/openai/multiagent-particle-envs.git
        if [ $? -ne 0 ]; then
            log_warn "GitHub克隆失败，尝试镜像..."
            git clone https://gitee.com/mirrors/multiagent-particle-envs.git
        fi
    fi
    
    if [ ! -d "multiagent-particle-envs" ]; then
        log_error "MPE克隆失败"
        return 1
    fi
    
    # 安装
    cd "multiagent-particle-envs"
    log_info "安装MPE..."
    pip install -e .
    cd ".."
    
    # 验证
    if python -c "import multiagent; print('MPE安装成功')" 2>/dev/null; then
        log_info "MPE安装验证成功"
    else
        log_warn "MPE安装可能有问题"
    fi
}

# 快速测试
run_quick_test() {
    log_info "运行快速测试..."
    
    echo
    log_info "测试算法框架..."
    python test_framework_simple.py
    
    echo
    log_info "运行示例实验..."
    python run_example.py
    
    log_info "快速测试完成!"
}

# 小规模实验
run_small_experiment() {
    log_info "运行小规模实验..."
    
    # 检查GPU
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        GPU_ARG="--gpu 0"
        log_info "检测到GPU，使用GPU模式"
    else
        GPU_ARG="--gpu -1"
        log_warn "未检测到GPU，使用CPU模式"
    fi
    
    OUTPUT_DIR="small_experiment_$(date +%Y%m%d_%H%M%S)"
    
    echo
    log_info "运行AdaptiveComm算法..."
    python experiments/main_experiment.py \
        $GPU_ARG \
        --env MPE_Navigation \
        --algorithms AdaptiveComm \
        --seeds 42 \
        --output_dir "$OUTPUT_DIR"
    
    log_info "小规模实验完成! 结果保存在: $OUTPUT_DIR"
}

# 完整论文实验
run_full_experiment() {
    log_info "运行完整论文实验..."
    
    # 检查GPU
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        GPU_ARG="--gpu 0"
        log_info "检测到GPU，使用GPU模式"
    else
        GPU_ARG="--gpu -1"
        log_warn "未检测到GPU，使用CPU模式"
    fi
    
    OUTPUT_DIR="full_experiment_$(date +%Y%m%d_%H%M%S)"
    
    echo
    log_info "运行完整实验 (5个算法，3个种子)..."
    python experiments/main_experiment.py \
        $GPU_ARG \
        --env MPE_Navigation \
        --algorithms MAPPO IACN SparseComm FullComm AdaptiveComm \
        --seeds 42 123 456 \
        --output_dir "$OUTPUT_DIR"
    
    log_info "完整实验完成! 结果保存在: $OUTPUT_DIR"
}

# 自定义实验
run_custom_experiment() {
    log_info "运行自定义实验..."
    
    # 获取用户输入
    echo
    read -p "输入环境 (默认: MPE_Navigation): " ENV
    ENV=${ENV:-MPE_Navigation}
    
    read -p "输入算法 (用空格分隔，默认: AdaptiveComm MAPPO): " ALGORITHMS
    ALGORITHMS=${ALGORITHMS:-AdaptiveComm MAPPO}
    
    read -p "输入随机种子 (用空格分隔，默认: 42 123): " SEEDS
    SEEDS=${SEEDS:-42 123}
    
    read -p "GPU ID (默认: 0，使用-1表示CPU): " GPU_ID
    GPU_ID=${GPU_ID:-0}
    
    OUTPUT_DIR="custom_experiment_$(date +%Y%m%d_%H%M%S)"
    
    echo
    log_info "实验配置:"
    echo "  环境: $ENV"
    echo "  算法: $ALGORITHMS"
    echo "  种子: $SEEDS"
    echo "  GPU: $GPU_ID"
    echo "  输出目录: $OUTPUT_DIR"
    echo
    
    read -p "是否开始实验? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "用户取消"
        exit 0
    fi
    
    # 运行实验
    python experiments/main_experiment.py \
        --gpu "$GPU_ID" \
        --env "$ENV" \
        --algorithms $ALGORITHMS \
        --seeds $SEEDS \
        --output_dir "$OUTPUT_DIR"
    
    log_info "自定义实验完成! 结果保存在: $OUTPUT_DIR"
}

# 运行主函数
main "$@"