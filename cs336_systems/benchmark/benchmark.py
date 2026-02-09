import argparse
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
    one_batch_data: str,
    warmup_iters: int,
    benchmark_iters: int,
    include_backward: bool,
    include_optimizer_step: bool,
    device: str = "cpu",
    use_nsys: bool = False,
) -> None:
    print("model device:", next(model.parameters()).device)
    print("x device:", one_batch_data[0].device)
    model = torch.compile(model)
    model.train() if include_backward else model.eval()
    optimizer = AdamW(model.parameters())

    # Warmup phase
    if use_nsys:
        nvtx.range_push("warmup")
    for _ in range(warmup_iters):
        x, _ = one_batch_data
        logits = model(x)
        if include_backward:
            loss = cross_entropy(logits, x)
            model.zero_grad(set_to_none=True)
            loss.backward()
            if include_optimizer_step:  
                optimizer.step()
        if device != "cpu":
            torch.cuda.synchronize()
    if use_nsys:
        nvtx.range_pop()

    # Benchmark phase
    times = []
    if use_nsys:
        nvtx.range_push("benchmark")
    for i in range(benchmark_iters):
        if use_nsys:
            nvtx.range_push(f"iteration_{i}")
        start = timeit.default_timer()
        x, _ = one_batch_data
        if use_nsys:
            nvtx.range_push("forward")
        logits = model(x)
        if use_nsys:
            nvtx.range_pop()
        if include_backward:
            loss = cross_entropy(logits, x)
            model.zero_grad(set_to_none=True)
            # 反向传播
            if use_nsys:
                nvtx.range_push("backward")
            loss.backward()
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
        end = timeit.default_timer()
        if use_nsys:
            nvtx.range_pop()
        print(f"Iteration time: {end - start:.4f} seconds")
        times.append(end - start)
    if use_nsys:
        nvtx.range_pop()

    # 计算 mean 和 std
    times = np.array(times)
    time_mean, time_std = np.mean(times), np.std(times)
    print(
        f"Average time over {benchmark_iters} iterations: {time_mean:.4f} seconds ± {time_std:.4f} seconds"
    )


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
    )


if __name__ == "__main__":
    print("Starting benchmark...")
    main()
