import argparse
from contextlib import nullcontext
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import cs336_basics
from cs336_basics.model.transformer_lm import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.optimizer import AdamW
from einops import einsum
from cs336_basics.model.attention import softmax
# Enable TensorFloat32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision("high")

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: torch.Tensor Query tensor of shape (..., q, d_k)
        K: torch.Tensor Key tensor of shape (..., k, d_k)
        V: torch.Tensor Value tensor of shape (..., k, d_v)
        mask: torch.Tensor | None Mask tensor of shape (q, k)

    Returns:
        torch.Tensor: The output tensor of shape (..., q, d_v)
    """
    d_k = Q.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_score = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / d_k**0.5
    if mask is not None:
        attention_score.masked_fill_(mask == 0, float("-inf"))
    with nvtx.range("computing softmax"):
        attn_weight = softmax(attention_score, dim=-1)
    with nvtx.range("final matmul"):
        output = einsum(attn_weight, V, "... q k, ... k d_v -> ... q d_v")
    return output

cs336_basics.model.attention.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def benchmark_model(
    model: nn.Module,
    one_batch_data: tuple[torch.Tensor, torch.Tensor],
    warmup_iters: int,
    benchmark_iters: int,
    include_backward: bool,
    include_optimizer_step: bool,
    device: str = "cpu",
    use_nsys: bool = False,
    use_amp: bool = False,
    use_memory_profiler: bool = False,
) -> None:
    model = torch.compile(model)
    model.train() if include_backward else model.eval()
    optimizer = AdamW(model.parameters())
    
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp and device != "cpu" else nullcontext()
    
    # Warmup phase
    if use_nsys:
        nvtx.range_push("warmup")
    for _ in range(warmup_iters):
        x, _ = one_batch_data
        # forward和loss计算需要在ctx下执行
        with ctx:
            logits = model(x)
            if include_backward:
                loss = cross_entropy(logits, x)
                model.zero_grad(set_to_none=True)
        if include_backward:
            loss.backward()
            if include_optimizer_step:  
                optimizer.step()
        if device != "cpu":
            torch.cuda.synchronize()
    if use_nsys:
        nvtx.range_pop()

    # Benchmark phase
    if use_memory_profiler:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    times = []
    if use_nsys:
        nvtx.range_push("benchmark")

    forward_time = []
    backward_time = []
    for i in range(benchmark_iters):
        if use_nsys:
            nvtx.range_push(f"iteration_{i}")
        x, _ = one_batch_data
        if use_nsys:
            nvtx.range_push("forward")
        # forward和loss计算需要在ctx下执行   
        with ctx:
            if device != "cpu":
                torch.cuda.synchronize()
            start = timeit.default_timer()
            logits = model(x)
            if device != "cpu":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_time.append(end - start)
            if use_nsys:
                nvtx.range_pop()
            if include_backward:
                loss = cross_entropy(logits, x)
        if include_backward:
            model.zero_grad()
            # 反向传播
            if use_nsys:
                nvtx.range_push("backward")
            if device != "cpu":
                torch.cuda.synchronize()
            start = timeit.default_timer()
            loss.backward()
            if device != "cpu":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            backward_time.append(end - start)
            if use_nsys:
                nvtx.range_pop()
            # optimizer step
            if include_optimizer_step: 
                if use_nsys:
                    nvtx.range_push("optimizer_step")  
                optimizer.step() 
                if use_nsys:
                    nvtx.range_pop()
            if device != "cpu":
                torch.cuda.synchronize()

        if use_nsys:
            nvtx.range_pop()
    if use_nsys:
        nvtx.range_pop()

    # 计算 mean 和 std
    forward_times = np.array(forward_time)
    backward_times = np.array(backward_time)
    print(f"Average forward time over {benchmark_iters} iterations: {forward_times.mean():.4f} seconds ± {forward_times.std():.4f} seconds")
    if include_backward:
        print(f"Average backward time over {benchmark_iters} iterations: {backward_times.mean():.4f} seconds ± {backward_times.std():.4f} seconds")

    if use_memory_profiler:
        torch.cuda.memory._dump_snapshot("cs336_systems/benchmark/logs/memory_snapshot_all_stage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Tool")
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument(
        "--d_model", type=int, default=768, help="Dimension of the model"
    )
    parser.add_argument(
        "--d_ff", type=int, default=3072, help="Dimension of the feedforward layer"
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of layers in the model"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    # Device and Data type
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    # data, 这是使用val data来进行benchmark
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="/home/jkd/online_course/cs336/cs336-assignment1/data/TinyStoriesV2-GPT4-valid_tokens.npy",
        help="Path to validation data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for benchmarking"
    )

    # warm-up iterations and benchmark iterations
    parser.add_argument(
        "--warmup_iters", type=int, default=5, help="Number of warm-up iterations"
    )
    parser.add_argument(
        "--benchmark_iters", type=int, default=10, help="Number of benchmark iterations"
    )

    # 是否包含反向传播
    parser.add_argument(
        "--include_backward",
        action="store_true",
        help="Include backward pass in benchmark",
    )

    # 是否包含更新optimizer
    parser.add_argument(
        "--include_optimizer_step",
        action="store_true",
        help="Include optimizer step in benchmark (only relevant if --include_backward is set)",
    )

    # 是否使用 nsys profiling
    parser.add_argument(
        "--use_nsys", action="store_true", help="Enable NVTX markers for nsys profiling"
    )

    # 是否使用amp
    parser.add_argument(
        "--use_amp", action="store_true", help="Use Automatic Mixed Precision (AMP) during benchmarking"
    )

    # 是否使用memory profiler
    parser.add_argument(
        "--use_memory_profiler", action="store_true", help="Enable memory profiling"
    )

    args = parser.parse_args()

    # 转换 dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        args.norm_eps,
        device=args.device,
        dtype=dtype,
    )
    val_data = np.load(args.val_data_path, mmap_mode="r")
    one_batch_data = get_batch(
        val_data, args.batch_size, args.context_length, args.device
    )
    benchmark_model(
        model,
        one_batch_data,
        args.warmup_iters,
        args.benchmark_iters,
        args.include_backward,
        args.include_optimizer_step,
        device=args.device,
        use_nsys=args.use_nsys,
        use_amp=args.use_amp,
        use_memory_profiler=args.use_memory_profiler,
    )


if __name__ == "__main__":
    print("Starting benchmark...")
    main()
