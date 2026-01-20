#!/bin/bash

# ============================================
# CentOS 7 CUDA/GPU修复脚本
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查命令
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# 检查GPU
check_gpu() {
    log_step "检查GPU硬件..."
    
    if check_command lspci; then
        if lspci | grep -i nvidia &> /dev/null; then
            log_info "检测到NVIDIA GPU"
            lspci | grep -i nvidia
            return 0
        else
            log_error "未检测到NVIDIA GPU"
            return 1
        fi
    else
        log_warn "无法检查GPU硬件 (lspci未安装)"
        return 2
    fi
}

# 检查NVIDIA驱动
check_nvidia_driver() {
    log_step "检查NVIDIA驱动..."
    
    if check_command nvidia-smi; then
        log_info "NVIDIA驱动已安装"
        nvidia-smi
        return 0
    else
        log_error "NVIDIA驱动未安装"
        return 1
    fi
}

# 检查CUDA
check_cuda() {
    log_step "检查CUDA工具包..."
    
    if check_command nvcc; then
        log_info "CUDA工具包已安装"
        nvcc --version
        return 0
    else
        log_error "CUDA工具包未安装"
        return 1
    fi
}

# 检查PyTorch CUDA
check_pytorch_cuda() {
    log_step "检查PyTorch CUDA支持..."
    
    python3 -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}:', torch.cuda.get_device_name(i))
"
}

# 安装NVIDIA驱动
install_nvidia_driver() {
    log_step "安装NVIDIA驱动..."
    
    log_info "添加ELRepo仓库..."
    sudo rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
    sudo yum install -y https://www.elrepo.org/elrepo-release-7.el7.elrepo.noarch.rpm
    
    log_info "安装驱动..."
    sudo yum install -y kmod-nvidia
    
    log_info "加载驱动模块..."
    sudo modprobe nvidia
    
    log_info "验证安装..."
    if check_command nvidia-smi; then
        log_info "NVIDIA驱动安装成功"
        nvidia-smi
    else
        log_error "驱动安装失败"
        return 1
    fi
    
    return 0
}

# 安装CUDA工具包
install_cuda_toolkit() {
    log_step "安装CUDA工具包..."
    
    # CentOS 7 CUDA安装
    log_info "下载CUDA安装包..."
    CUDA_VERSION="11.8"
    CUDA_RPM="cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm"
    
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/${CUDA_RPM}
    
    log_info "安装CUDA..."
    sudo rpm -i ${CUDA_RPM}
    sudo yum clean all
    sudo yum install -y cuda
    
    log_info "设置环境变量..."
    echo "export PATH=/usr/local/cuda/bin:\$PATH" | sudo tee -a /etc/profile.d/cuda.sh
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" | sudo tee -a /etc/profile.d/cuda.sh
    source /etc/profile.d/cuda.sh
    
    log_info "验证安装..."
    if check_command nvcc; then
        log_info "CUDA安装成功"
        nvcc --version
    else
        log_error "CUDA安装失败"
        return 1
    fi
    
    return 0
}

# 重新安装PyTorch CUDA版本
reinstall_pytorch_cuda() {
    log_step "重新安装PyTorch CUDA版本..."
    
    # 检查当前CUDA版本
    if check_command nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        
        log_info "检测到CUDA ${CUDA_VERSION}"
        
        case $CUDA_MAJOR in
            11)
                TORCH_URL="https://download.pytorch.org/whl/cu118"
                ;;
            12)
                TORCH_URL="https://download.pytorch.org/whl/cu121"
                ;;
            *)
                log_warn "不支持的CUDA版本，使用CUDA 11.8"
                TORCH_URL="https://download.pytorch.org/whl/cu118"
                ;;
        esac
        
        log_info "安装PyTorch for CUDA ${CUDA_MAJOR}.x..."
        pip uninstall -y torch torchvision torchaudio
        pip install torch torchvision torchaudio --index-url ${TORCH_URL}
        
    else
        log_warn "未检测到CUDA，安装PyTorch CPU版本"
        pip uninstall -y torch torchvision torchaudio
        pip install torch torchvision torchaudio
    fi
}

# 测试GPU
test_gpu() {
    log_step "测试GPU功能..."
    
    python3 -c "
import torch
print('='*50)
print('GPU测试')
print('='*50)

if torch.cuda.is_available():
    print('✅ CUDA可用')
    print(f'GPU数量: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        print(f'\nGPU {i}:')
        print(f'  名称: {torch.cuda.get_device_name(i)}')
        print(f'  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
    
    # 性能测试
    print('\n性能测试:')
    import time
    
    # 创建大张量
    size = 5000
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    # GPU计算
    start = time.time()
    c = torch.matmul(a, b)
    gpu_time = time.time() - start
    
    print(f'  GPU矩阵乘法 ({size}x{size}): {gpu_time:.3f} 秒')
    
    # 清理
    del a, b, c
    torch.cuda.empty_cache()
    
    print('\n✅ GPU测试通过')
else:
    print('❌ CUDA不可用')
"
}

# 主函数
main() {
    echo -e "${BLUE}""=" 80""${NC}"
    echo -e "${BLUE}CentOS 7 GPU/CUDA修复脚本${NC}"
    echo -e "${BLUE}""=" 80""${NC}"
    echo
    
    # 检查当前用户
    if [[ $EUID -eq 0 ]]; then
        log_warn "请不要使用root用户运行此脚本"
        log_warn "请使用普通用户运行，脚本会在需要时请求sudo权限"
        exit 1
    fi
    
    # 1. 检查GPU硬件
    if ! check_gpu; then
        log_error "服务器可能没有NVIDIA GPU，无法使用CUDA"
        log_info "可以考虑:"
        log_info "  1. 使用CPU模式运行实验 (较慢)"
        log_info "  2. 申请带GPU的服务器"
        log_info "  3. 使用云GPU服务"
        exit 1
    fi
    
    # 2. 检查NVIDIA驱动
    if ! check_nvidia_driver; then
        log_warn "NVIDIA驱动未安装"
        read -p "是否安装NVIDIA驱动? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_nvidia_driver
        else
            log_error "必须安装NVIDIA驱动才能使用GPU"
            exit 1
        fi
    fi
    
    # 3. 检查CUDA工具包
    if ! check_cuda; then
        log_warn "CUDA工具包未安装"
        read -p "是否安装CUDA工具包? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_cuda_toolkit
        else
            log_error "必须安装CUDA工具包才能使用GPU"
            exit 1
        fi
    fi
    
    # 4. 检查PyTorch CUDA
    log_info "当前PyTorch状态:"
    check_pytorch_cuda
    
    # 5. 重新安装PyTorch CUDA版本
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log_warn "PyTorch CUDA不可用"
        read -p "是否重新安装PyTorch CUDA版本? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            reinstall_pytorch_cuda
        fi
    fi
    
    # 6. 测试GPU
    test_gpu
    
    # 7. 运行实验的建议
    echo
    log_step "运行实验建议:"
    echo -e "${GREEN}""=" 40""${NC}"
    
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log_info "✅ GPU已就绪，可以运行实验"
        echo
        log_info "运行实验命令:"
        echo "cd /home/xxjss/code/xu_first"
        echo "export SUPPRESS_MA_PROMPT=1"
        echo "python experiments/main_experiment.py --gpu 0 --env MPE_Navigation --algorithms AdaptiveComm"
    else
        log_error "❌ GPU仍然不可用"
        echo
        log_info "备选方案:"
        echo "1. 使用CPU模式: --gpu -1"
        echo "2. 联系系统管理员检查GPU"
        echo "3. 使用云GPU服务"
    fi
    
    echo -e "${GREEN}""=" 40""${NC}"
    echo
    log_info "脚本执行完成"
}

# 运行主函数
main "$@"