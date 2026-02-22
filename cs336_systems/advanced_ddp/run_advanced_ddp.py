import argparse
import os
import random
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import torch.cuda.nvtx as nvtx
from cs336_basics.model.transformer_lm import TransformerLM
from functools import partial
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_systems.advanced_ddp.ddp import DDP, DDPWithBucket
# 构建闭包
def build_model(model_name, model_kwargs):
    if model_name == "ToyModel":
        return ToyModel()
    if model_name == "TransformerLM":
        return TransformerLM(**model_kwargs)
    raise ValueError(f"Unknown model: {model_name}")


from tests.common import (
    ToyModel,
    _setup_process_group,
    _cleanup_process_group
)


def seed_everything(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def shard_batch(x, y, rank, world_size):
    shard_size = x.shape[0] // world_size
    start_idx = shard_size * rank
    end_idx = start_idx + shard_size
    return x[start_idx:end_idx], y[start_idx:end_idx]

def run_naive_ddp(rank, world_size, backend, model_fn, num_steps, val_data, batch_size, context_length,
                  base_seed, deterministic, use_nsys, bucket_size_mb):
    device = _setup_process_group(rank, world_size, backend=backend)
    seed_everything(base_seed, deterministic=deterministic)
    model = model_fn().to(device)
    if bucket_size_mb:
        model = DDPWithBucket(model, bucket_size_mb=bucket_size_mb)
    else:
        model = DDP(model)
    optimizer = AdamW(model.parameters())

    # warm up
    print(f"Rank {rank} starting warmup...")
    for step in range(5):
        # 产生数据
        x, y = get_batch(
            val_data, batch_size, context_length, device
        )
        shard_x, shard_y = shard_batch(x, y, rank, world_size)
        # 开始训练
        pred = model(shard_x)
        loss = cross_entropy(pred, shard_y)
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss /= world_size
        optimizer.zero_grad()
        loss.backward()

        # 同步梯度
        model.finish_gradient_synchronization()

        # 更新参数
        optimizer.step()
    if backend == "nccl":
        torch.cuda.synchronize()
    print(f"Rank {rank} finished warmup.")

    # benchmark
    for step in range(num_steps):
        if use_nsys:
            nvtx.range_push(f"step_{step}")
        if backend == "nccl":
            torch.cuda.synchronize()
        step_start_time = time.perf_counter()
        # 产生数据
        x, y = get_batch(
            val_data, batch_size, context_length, device
        )
        shard_x, shard_y = shard_batch(x, y, rank, world_size)
        # 开始训练
        if use_nsys:
            nvtx.range_push("forward")
        pred = model(shard_x)
        if use_nsys:
            nvtx.range_pop()
        loss = cross_entropy(pred, shard_y)
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss /= world_size
        if rank == 0:
            print(f"Step {step}, Global Mean Loss: {global_loss.item()}")
        optimizer.zero_grad()
        if use_nsys:
            nvtx.range_push("backward")
        loss.backward()
        if use_nsys:
            nvtx.range_pop()

        if backend == "nccl":
            torch.cuda.synchronize()
            start = time.perf_counter()
        
        # 同步梯度
        if use_nsys:
            nvtx.range_push("gradient_sync")
        model.finish_gradient_synchronization()
        if use_nsys:
            nvtx.range_pop()
        if backend == "nccl":
            torch.cuda.synchronize()
        end = time.perf_counter()
        local_time = end - start
        global_time = torch.tensor(local_time, device=device)
        dist.all_reduce(global_time, op=dist.ReduceOp.AVG)
        # 更新参数
        optimizer.step()
        if backend == "nccl":
            torch.cuda.synchronize()
        step_end_time = time.perf_counter()
        step_local_time = step_end_time - step_start_time
        step_global_time = torch.tensor(step_local_time, device=device)
        dist.all_reduce(step_global_time, op=dist.ReduceOp.AVG)
        
        if rank == 0:
            print(f"Step {step}, Avg Step Time: {step_global_time.item():.4f} seconds, Avg Commuication Time: {global_time.item():.4f} seconds, Ration: {global_time.item() / step_global_time.item():.4f}")
        # 结束这个step的nvtx range
        if use_nsys:
            nvtx.range_pop()
    if rank == 0:
        # 保存这个rank的模型参数
        save_path = f"cs336_systems/naive_ddp/checkpoints/naive_ddp_model_{world_size}_process.pt"
        torch.save(model.state_dict(), save_path)

    _cleanup_process_group()

def main():
    parser = argparse.ArgumentParser(description="Run naive DDP.")
    parser.add_argument(
        "--world_size", type=int, default=2, help="Total number of processes."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="Distributed backend to use (e.g., gloo, nccl).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ToyModel",
        help="Model to use (e.g., ToyModel).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps to train for."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. All ranks use this seed."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch algorithms (may reduce performance)."
    )
    # data, 这是使用val data来进行benchmark
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="/root/cs336/cs336-assignment1/data/TinyStoriesV2-GPT4-valid_tokens.npy",
        help="Path to validation data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=128
    )
    parser.add_argument(
        "--use_nsys",
        action="store_true",
    )
    parser.add_argument(
        "--bucket_size_mb",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    val_data = np.load(args.val_data_path, mmap_mode="r")
    
    model_fn = partial(build_model, model_name=args.model, model_kwargs=dict(
        vocab_size=10000,
        context_length=args.context_length,
        # small
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,

        # medium
        # d_model=1024,
        # num_heads=16,
        # num_layers=24,
        # d_ff=4096,

        # large
        # d_model=1280,
        # num_heads=20,
        # num_layers=36,
        # d_ff=5120,

        # xl
        # d_model=1600,
        # num_heads=25,
        # num_layers=48,
        # d_ff=6400,
    ))

    mp.spawn(
        run_naive_ddp,
        args=(args.world_size, args.backend, model_fn, args.num_steps, val_data, 
              args.batch_size, args.context_length, args.seed, args.deterministic, 
              args.use_nsys, args.bucket_size_mb),
        nprocs=args.world_size,
        join=True,
    )

if __name__ == "__main__":
    main()