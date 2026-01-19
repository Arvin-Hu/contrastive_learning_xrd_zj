#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
export CUDA_VISIBLE_DEVICES=0

NPROC_PER_NODE=1

# 当前节点编号：单节点场景下为0
NODE_RANK=0

# 总进程数（可选，通常是 NNODES * NPROC_PER_NODE）
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --rdzv_conf timeout=6000 \
         train_crystalsystem.py \
         --learning_rate 1e-4 \
         --epochs 60 \
         --batch_size 512 \
         --embedding_dim 256 \
         --output_path /mnt/minio/battery/xrd/train_outputs/xrd/crystal_system/v1 \
         --train_path /mnt/minio/battery/xrd/datasets/MP_crystalsystem_QA_train.jsonl  \
         --eval_path /mnt/minio/battery/xrd/datasets/MP_crystalsystem_QA_test.jsonl


# --model_path /home/perm/workspace/data/mnt/minio/battery/xrd/train_outputs/xrd/crystal_system/epoch_1.pth

# /mnt/minio/battery/xrd/train_outputs/xrd/formation_energy/yyh



