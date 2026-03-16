#!/bin/bash

#SBATCH --job-name=FlowRL_qwen3_8B_math_train
#SBATCH -N 2
#SBATCH --ntasks-per-node=1           # 改为1
#SBATCH --cpus-per-task=64            # 添加CPU核心数        
#SBATCH --gres=gpu:8                  
#SBATCH --mem=960G
#SBATCH --partition=gpu    # 明确指定目标队列
#SBATCH --qos=high
#SBATCH --mail-type=end
#SBATCH --mail-user=someone@example.org
#SBATCH --output=./logs/%x_%j/%j.out
#SBATCH --error=./logs/%x_%j/%j.err

module purge
module load cuda/12.8

# source /data/home/ytyuan/anaconda3/bin/activate /data/home/ytyuan/anaconda3/envs/verl

# 清除Ray环境变量
unset RAY_ADDRESS
unset RAY_CLUSTER_NAME
unset RAY_HEAD_NODE_HOST
export RAY_ADDRESS=""

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

# 這裡取消設定衝突的 AMD GPU 環境變數
unset ROCR_VISIBLE_DEVICES

# ===============================================
# 1. 关键：多机网络配置 (参考管理员提供的方案)
# ===============================================
# # 必须加上这些，否则多机通信可能会 hang 住
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1

# -----------------------------------------------
# A. 启用 InfiniBand (IB) - 提速核心
# -----------------------------------------------
export NCCL_IB_DISABLE=0                # 0 = 启用 IB
export NCCL_IB_HCA=mlx5_200g_0:1,mlx5_200g_1:1,mlx5_200g_2:1,mlx5_200g_3:1    # 官方指定的 IB 网卡设备号
export NCCL_IB_TIMEOUT=23               # 官方建议参数
export NCCL_IB_RETRY_CNT=7              # 官方建议参数

# -----------------------------------------------
# B. 保留以太网作为“握手”和“辅助”通道
# -----------------------------------------------
# Socket (TCP) 依然走 bond0，用于 Ray 管理通信和 NCCL 握手
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
# 强制 NCCL 和 Gloo 仅使用 IPv4 协议，防止 v4/v6 混用导致的握手失败
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
# -----------------------------------------------
# C. 其他 NCCL 优化参数
# -----------------------------------------------
export NCCL_DEBUG=INFO
# 启用 P2P (通常 IB 开启后，P2P 也应该开启，除非集群明确不支持)
# 如果遇到显存总线错误，可尝试设为 1，但优先设为 0 或 unset
export NCCL_P2P_DISABLE=0 
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# ===============================================
# 2. 关键：在 Slurm 中启动 Ray 集群
# ===============================================
echo "=== Starting Ray Cluster ==="

# 2.0 [新增] 强力清理残留进程
# 必须先清理环境，防止端口冲突 (ConnectionError) 和 /tmp/ray 权限问题
echo "Cleaning up old Ray processes and temp files on all nodes..."
# 忽略报错 (|| true)，因为如果没有残留进程 killall 会报错
srun --overlap --nodes=${SLURM_NNODES} --ntasks-per-node=1 bash -c "
    killall -9 ray 2>/dev/null || true
    # 关键：清除 /tmp/ray 下可能有权限问题的残留文件
    rm -f /tmp/ray/ray_current_cluster 2>/dev/null || true
    rm -rf /tmp/ray/session_* 2>/dev/null || true
" || true
# 等待几秒确保端口释放
sleep 5

# 2.1 获取节点信息
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Head Node: $head_node"

# 获取 Head 节点 IP
head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head Node IP: $head_node_ip"
export RAY_PORT=6379

LARGE_N=99999999999
export RAY_health_check_initial_delay_ms=$LARGE_N
export RAY_health_check_period_ms=$LARGE_N

# 2.2 启动 Ray Head (在第一个节点)
# 注意：这里我们启动 Head，并指定 Dashboard 端口为 8265，但核心通信端口是 6379
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; exec ray start --head --node-ip-address=$head_node_ip --port=$RAY_PORT --num-gpus=8 --dashboard-host=0.0.0.0 --temp-dir=/tmp/ray_${SLURM_JOB_ID} --block" &

# 等待 Head 启动
sleep 10

# 2.3 启动 Ray Worker (在剩余节点)
# 注意：Worker 连接的是 HeadIP:6379
for ((i=1; i<${#nodes_array[@]}; i++)); do
    worker_node=${nodes_array[$i]}
    echo "Starting Ray Worker on $worker_node..."
    srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
        bash -c "unset ROCR_VISIBLE_DEVICES; exec ray start --address=$head_node_ip:$RAY_PORT --num-gpus=8 --temp-dir=/tmp/ray_${SLURM_JOB_ID} --block" &
done

# 等待 Worker 连接
sleep 10

# 2.4 设置环境变量供 Python 脚本使用
export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"
export NNODES=${#nodes_array[@]}  # 自动计算节点数传递给脚本

echo "Ray Cluster Status:"
# 检查集群状态，应该显示 16 张卡
ray status

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


# 1. Increase Ray Timeout Thresholds
export RAY_health_check_timeout_ms=1200000  # 20 minutes
export RAY_keep_alive_timeout_ms=1200000
# 2. Increase NCCL Timeout
export NCCL_TIMEOUT=7200

# # 禁用 Triton/Inductor 的本地文件缓存机制
# export TORCHINDUCTOR_FX_GRAPH_CACHE=0
# export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# export MAX_JOBS=72
export NCCL_DEBUG=INFO

# 运行脚本前先在日志中打印脚本
# script_path="./examples/grpo_trainer/run_qwen3moe-30b_megatron_96gb.sh"
script_path="./recipe/flowrl/run_flowrl_qwen3_8b_megatron.sh"

echo "=== Running Training Script on Head Node: $head_node ==="
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES; export NCCL_SOCKET_FAMILY=AF_INET; export GLOO_SOCKET_FAMILY=AF_INET; export NCCL_TIMEOUT=7200; bash $script_path"