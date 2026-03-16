#!/bin/bash

#SBATCH --job-name=FlowRL_qwen1.5_A2.7B_math_train_megatron
#SBATCH -N 1
#SBATCH --ntasks-per-node=1           # 改为1
#SBATCH --cpus-per-task=96            # 添加CPU核心数
#SBATCH --gres=gpu:8                  # 改为4个GPU
#SBATCH --mem=900G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=someone@example.org
#SBATCH --output=./slurm/logs/%x_%j/%j.out
#SBATCH --error=./slurm/logs/%x_%j/%j.err

# # 清除Ray环境变量
# unset RAY_ADDRESS
# unset RAY_CLUSTER_NAME
# unset RAY_HEAD_NODE_HOST
# export RAY_ADDRESS=""

# 打印调试信息
echo "=== SLURM Resource Allocation ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
nvidia-smi

echo "=== Starting Training ==="

export WANDB_API_KEY="047f8e7060103f352673be2eeb55df2d1996dfc3"

export https_proxy=http://10.80.80.7:7890
export http_proxy=http://10.80.80.7:7890
export all_proxy=socks://10.80.80.7:7890
export WANDB_MODE=offline
# # # 2. 告诉 W&B 我们要恢复一个实验
# export WANDB_RESUME=allow
# # # 3. 指定要恢复的旧实验ID
# export WANDB_ID=64de032n
# 這裡取消設定衝突的 AMD GPU 環境變數
unset ROCR_VISIBLE_DEVICES

# ===============================================
# 环境变量配置：本地CUDA + Conda cuDNN (Symlink Fix)
# -----------------------------------------------

export CUDA_HOME=/usr/local/cuda

# 1. 获取 pip 安装的 nvidia-cudnn 的路径
export CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")

# 2. 告诉 CMake 和 编译器 头文件在哪里
export CUDNN_INCLUDE_DIR="${CUDNN_PATH}/include"
export CPATH="${CUDNN_PATH}/include:${CPATH}"

# 3. 告诉 链接器 库文件在哪里
export CUDNN_LIBRARY="${CUDNN_PATH}/lib"
export LIBRARY_PATH="${CUDNN_PATH}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH}"

# 4. 打印一下确认路径是否正确 (可选)
echo "CUDNN Path found at: $CUDNN_PATH"



# # 1. Increase Ray Timeout Thresholds
export RAY_health_check_timeout_ms=1200000  # 20 minutes
export RAY_keep_alive_timeout_ms=1200000
# 2. Increase NCCL Timeout
export NCCL_TIMEOUT=7200

# 禁用 Triton/Inductor 的本地文件缓存机制
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

export TORCH_COMPILE_DISABLE=1

export MAX_JOBS=72

export NCCL_DEBUG=INFO

export RAY_ADDRESS="10.80.80.16:6379"  # 确保显式指定 IP
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0

export NCCL_IB_DISABLE=1

# 运行脚本前先在日志中打印脚本
script_path="./recipe/flowrl/run_flowrl_qwen3_30b.sh"

bash "$script_path"