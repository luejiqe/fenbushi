#!/bin/bash
# DeepSpeed Training Launch Script
# 使用DeepSpeed ZeRO-2启动训练

# 基本用法
# bash train_deepspeed.sh

# 自定义GPU数量
# bash train_deepspeed.sh --num_gpus=2

# 配置参数
TRAIN_DIR="data/train"
VAL_DIR="data/val"
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
NUM_WORKERS=2

# DeepSpeed参数
NUM_GPUS=1
FP16=""
OFFLOAD_OPTIMIZER=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --offload_optimizer)
            OFFLOAD_OPTIMIZER="--offload_optimizer"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "DeepSpeed ZeRO-2 Training Launch"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Batch Size (per GPU): $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "FP16: ${FP16:-Disabled}"
echo "Offload Optimizer: ${OFFLOAD_OPTIMIZER:-Disabled}"
echo "========================================"

# 启动DeepSpeed训练
deepspeed --num_gpus=$NUM_GPUS src/train_deepspeed.py \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    $FP16 \
    $OFFLOAD_OPTIMIZER
