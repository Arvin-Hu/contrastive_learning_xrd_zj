#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
export CUDA_VISIBLE_DEVICES=2

NPROC_PER_NODE=1

# 当前节点编号：单节点场景下为0
NODE_RANK=0

# 总进程数（可选，通常是 NNODES * NPROC_PER_NODE）
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))


#TODO early stop; 保存best ckpt; 训练数据异常点；加入化学式；数据增强策略


# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --rdzv_conf timeout=6000 \
         train_reg.py \
         --learning_rate 1e-5 \
         --epochs 100 \
         --batch_size 32 \
         --embedding_dim 256 \
         --output_path /mnt/minio/battery/xrd/train_outputs/xrd/formation_energy \
         --train_path /mnt/minio/battery/xrd/datasets/MP_formationenergy-QA-train.jsonl  \
         --eval_path /mnt/minio/battery/xrd/datasets/MP_formationenergy-QA-test.jsonl
 
         
        
         
         