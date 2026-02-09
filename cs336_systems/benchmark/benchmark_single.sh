#!/bin/bash

# 默认配置
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=768
D_FF=3072
NUM_LAYERS=12
NUM_HEADS=12
ROPE_THETA=10000.0
NORM_EPS=1e-5

DEVICE="cuda:0"
DTYPE="float32"

VAL_DATA_PATH="../cs336-assignment1/data/TinyStoriesV2-GPT4-valid_tokens.npy"
BATCH_SIZE=4

WARMUP_ITERS=5
BENCHMARK_ITERS=10

# 运行前向传播 benchmark
echo "========== Forward Pass Benchmark =========="
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
    --benchmark_iters $BENCHMARK_ITERS

# 运行前向+反向传播 benchmark
echo ""
echo "========== Forward + Backward Pass Benchmark =========="
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
    --include_backward