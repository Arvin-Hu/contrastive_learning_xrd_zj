#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
export CUDA_VISIBLE_DEVICES=3

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
         train.py \
         --use_qformer True \
         --learning_rate 1e-4 \
         --epochs 400 \
         --projection_dim 64 \
         --batch_size 32 \
         --temperature 0.1 \
         --embedding_dim 256 \
         --model_path /mnt/minio/battery/xrd/train_outputs/contrastive_learning/peak_v0/epoch_83.pth \
         --output_path /mnt/minio/battery/xrd/train_outputs/contrastive_learning/peak_v0
         
         
        
         
         