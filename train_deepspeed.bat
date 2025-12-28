@echo off
REM DeepSpeed Training Launch Script for Windows
REM 使用DeepSpeed ZeRO-2启动训练

REM 配置参数
set TRAIN_DIR=data/train
set VAL_DIR=data/val
set EPOCHS=50
set BATCH_SIZE=16
set LEARNING_RATE=0.001
set NUM_WORKERS=2

REM DeepSpeed参数
set NUM_GPUS=1
set FP16=
set OFFLOAD_OPTIMIZER=

echo ========================================
echo DeepSpeed ZeRO-2 Training Launch
echo ========================================
echo GPUs: %NUM_GPUS%
echo Batch Size (per GPU): %BATCH_SIZE%
echo Epochs: %EPOCHS%
echo ========================================

REM 启动DeepSpeed训练
deepspeed --num_gpus=%NUM_GPUS% src/train_deepspeed.py ^
    --train_dir %TRAIN_DIR% ^
    --val_dir %VAL_DIR% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --learning_rate %LEARNING_RATE% ^
    --num_workers %NUM_WORKERS% ^
    %FP16% ^
    %OFFLOAD_OPTIMIZER%
