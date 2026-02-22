uv run python -m cs336_systems.optmizer_shard.run_advanced_ddp_with_sharded_optim  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2 \
    --bucket_size_mb 10

uv run python -m cs336_systems.optmizer_shard.run_advanced_ddp_with_sharded_optim  \
    --model TransformerLM \
    --backend nccl \
    --world_size 2 \
    --bucket_size_mb 10 \
    --use_sharded_optimizer