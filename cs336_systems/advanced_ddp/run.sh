uv run python -m cs336_systems.advanced_ddp.run_advanced_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 1

uv run python -m cs336_systems.advanced_ddp.run_advanced_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2

uv run nsys profile -f true -o cs336_systems/advanced_ddp/res/result_nccl_2_process \
    python -m cs336_systems.advanced_ddp.run_advanced_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2 \
    --use_nsys

uv run python -m cs336_systems.advanced_ddp.run_advanced_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2 \
    --bucket_size_mb 10