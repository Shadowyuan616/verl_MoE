#!/bin/bash
set -x 

BACKEND="fsdp"  
LOCAL_DIR="/data/home/ytyuan/code/verl_MoE/ckpts/FlowRL/DAPO-FlowRL-Qwen3-30B-MATH-fsdp-Node2_bs256_1441_4_minbs64/global_step_150/actor/"
TARGET_DIR="/data/home/ytyuan/code/verl_MoE/ckpts/merged/0201DAPO-FlowRL-Qwen3-30B-MATH-fsdp-Node2_bs256_1441_4_minbs64/global_step_150"

PYTHONPATH=. python scripts/model_merger.py merge \
  --backend $BACKEND \
  --local_dir $LOCAL_DIR \
  --target_dir $TARGET_DIR

