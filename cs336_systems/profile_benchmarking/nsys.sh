#!/bin/bash

# 固定配置
VOCAB_SIZE=10000
CONTEXT_LENGTH=128
ROPE_THETA=10000.0
NORM_EPS=1e-5
DEVICE="cuda:0"
DTYPE="float32"
VAL_DATA_PATH="../cs336-assignment1/data/TinyStoriesV2-GPT4-valid_tokens.npy"
BATCH_SIZE=4
WARMUP_ITERS=5
BENCHMARK_ITERS=10

# 模型配置数组 (d_model d_ff num_layers num_heads)
CONFIGS=(
    "768 3072 12 12"      # small
)

NAMES=("small" "medium" "large" "xl")

# 遍历每种配置
for i in "${!CONFIGS[@]}"; do
    NAME="${NAMES[$i]}"
    read -r D_MODEL D_FF NUM_LAYERS NUM_HEADS <<< "${CONFIGS[$i]}"
    
    echo ""
    echo "========================================"
    echo "Model: $NAME"
    echo "d_model=$D_MODEL, d_ff=$D_FF, layers=$NUM_LAYERS, heads=$NUM_HEADS"
    echo "========================================"
    
    # 前向传播
    echo "----- Forward Pass -----"
    uv run nsys profile --trace=cuda,nvtx,osrt -f true -o cs336_systems/benchmark/logs/result_ctx128_$NAME \
        python cs336_systems/benchmark/benchmark.py \
        --vocab_size $VOCAB_SIZE \
        --context_length $CONTEXT_LENGTH \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --num_layers $NUM_LAYERS \
        --num_heads $NUM_HEADS \
        --rope_theta $ROPE_THETA \
        --norm_eps $NORM_EPS \
        --device $DEVICE \
        --dtype $DTYPE \
        --val_data_path $VAL_DATA_PATH \
        --batch_size $BATCH_SIZE \
        --warmup_iters $WARMUP_ITERS \
        --benchmark_iters $BENCHMARK_ITERS \
        --use_nsys
    
    echo ""
    echo "========================================"
    echo "Completed: $NAME"
    echo "========================================"

    # 分析配置，将结果保存到文本文件
    # echo "Context Length: $CONTEXT_LENGTH, Model: $NAME" >> cs336_systems/benchmark/logs/nvtx.log
    nsys stats --report cuda_gpu_kern_sum --timeunit s --force-export=true cs336_systems/benchmark/logs/result_ctx128_$NAME.nsys-rep
    # echo "=========================================" >> cs336_systems/benchmark/logs/nvtx.log
done