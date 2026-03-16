
BASE_DIR="<YOUR_BASE_PATH>"
# MODEL_PATH="/data3/ytyuan/checkpoints/FlowRL/RoutingReplay/merged/DAPO-GRPO-Qwen2.5-3B-Instruct-MATH-fsdp-Node1_bs256_1441_2_minbs64_micro_bs4/global_step_150"
# OUTPUT_DIR="/home/ytyuan/code/FlowRL/verl_Test/test_result/math/DAPO-GRPO-Qwen2.5-3B-Instruct-MATH-fsdp-Node1_bs256_1441_2_minbs64_micro_bs4/global_step_150"
MODEL_PATH="/data3/ytyuan/model/Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR="/home/ytyuan/code/FlowRL/verl_Test/test_result/math/DAPO-BASE-Qwen2.5-3B-Instruct-MATH"
DATA_PATH="/home/ytyuan/code/FlowRL/data/math_data"

n_gpus_per_node=5

# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    data.path=$DATA_PATH/aime-2024.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=16 \
    data.output_path=$OUTPUT_DIR/test-output-16.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m recipe.r1.main_eval \
    data.path=$OUTPUT_DIR/test-output-16.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func