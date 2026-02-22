uv run python -m cs336_systems.naive_ddp.run_naive_ddp  \
    --model ToyModel \
    --backend nccl \
    --world_size 1

uv run python -m cs336_systems.naive_ddp.run_naive_ddp  \
    --model ToyModel \
    --backend nccl \
    --world_size 2

uv run python -m cs336_systems.naive_ddp.run_naive_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 1

uv run python -m cs336_systems.naive_ddp.run_naive_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2

uv run nsys profile -f true -o cs336_systems/naive_ddp/res/result_nccl_2_process_naive \
    python -m cs336_systems.naive_ddp.run_naive_ddp  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2 \
    --use_nsys