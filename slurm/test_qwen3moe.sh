#!/bin/bash

#SBATCH --job-name=FlowRL_qwen3_A3B_math_train
#SBATCH -N 2
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=80            
#SBATCH --gres=gpu:8                  
#SBATCH --mem=700G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=someone@example.org
#SBATCH --output=./logs/%x_%j/%j.out
#SBATCH --error=./logs/%x_%j/%j.err

# ===============================================
# 1. 环境准备 (借鉴文档风格)
# ===============================================
# 加载文档推荐的模块，这通常包含IB驱动所需的库文件
# module load miniconda/24.11.1
# module load cuda/12.8
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


# 激活你的 conda 环境
source /data/home/ytyuan/anaconda3/bin/activate verl
# 打印一下 python 位置，确保加载正确
echo "Using Python: $(which python)"
echo "Conda Env: $CONDA_DEFAULT_ENV"

# 清理潜在的干扰变量
unset RAY_ADDRESS
unset ROCR_VISIBLE_DEVICES
export RAY_ADDRESS=""

# ===============================================
# 2. 获取节点列表 (借鉴文档风格)
# ===============================================
# 文档使用 for 循环手动构建数组，这里还原该逻辑
echo "=== Hostname Resolution ==="
declare -a nodes_array
k=0
for i in `scontrol show hostnames $SLURM_JOB_NODELIST`
do
    nodes_array[$k]=$i
    echo "Node $k: ${nodes_array[$k]}"
    let k=k+1
done

# 定义主节点 (Head) 和 工作节点 (Worker)
head_node=${nodes_array[0]}
echo "Head Node: $head_node"

# [关键修复] 获取 Head 节点的 IPv4 地址
# 在 sbatch 环境中，直接 hostname 可能会返回 IPv6 地址，导致 NCCL 握手失败 (wrong type 3!=4)
# 这里使用 getent hosts 强制获取 IPv4
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "getent hosts $head_node | awk '{print \$1}' | head -n 1")
echo "Head Node IP (IPv4): $head_node_ip"

# ===============================================
# 3. 网络配置 (文档+修复)
# ===============================================
# --- 文档推荐的 IB 配置 ---
export NCCL_IB_HCA=mlx5_6:1,mlx5_7:1
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# --- 强制绑定 TCP 接口 (解决握手失败) ---
# Ray/Gloo/NCCL 全部走 bond0 进行控制通信
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export RAY_node_ip_address=$head_node_ip 

# --- 强制 IPv4 协议 (解决 3!=4 报错) ---
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# --- 调试信息 ---
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# ===============================================
# 4. 启动 Ray 集群
# ===============================================
export RAY_PORT=6379
echo "=== Starting Ray Cluster ==="

# 4.1 清理旧进程
srun --overlap --nodes=${SLURM_NNODES} --ntasks-per-node=1 bash -c "killall -9 ray python" || true
sleep 5

# 4.2 启动 Head 节点
# 注意：关闭 Dashboard 以减少干扰
echo "Starting Ray Head on $head_node ($head_node_ip)..."
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; exec ray start --head \
    --node-ip-address=$head_node_ip --port=$RAY_PORT \
    --num-gpus=8 --include-dashboard=False --block" &

# 等待 Head 启动
sleep 15

# 4.3 启动 Worker 节点
# 遍历数组中除了第一个节点以外的所有节点
for ((i=1; i<${#nodes_array[@]}; i++)); do
    worker_node=${nodes_array[$i]}
    echo "Starting Ray Worker on $worker_node..."
    srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
        bash -c "unset ROCR_VISIBLE_DEVICES; exec ray start \
        --address=$head_node_ip:$RAY_PORT \
        --num-gpus=8 --include-dashboard=False --block" &
done

# 等待 Worker 连接
sleep 15

# 设置 Ray 地址供后续 Python 脚本使用
export RAY_ADDRESS="$head_node_ip:$RAY_PORT"
export NNODES=${#nodes_array[@]}

echo "Ray Cluster Status:"
ray status

# ===============================================
# 5. 运行训练脚本
# ===============================================
# 你的训练脚本路径
script_path="./recipe/flowrl/run_flowrl_qwen3_30b.sh"

# 其他环境变量
export WANDB_API_KEY="047f8e7060103f352673be2eeb55df2d1996dfc3"
export WANDB_MODE=offline
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAX_JOBS=72
# 增加超时防止初始化被杀
export RAY_health_check_timeout_ms=3600000 
export RAY_keep_alive_timeout_ms=3600000

echo "=== Running Training Script on Head Node: $head_node ==="
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; bash $script_path"