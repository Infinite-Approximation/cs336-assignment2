import argparse
import torch
import torch.nn as nn
import time
from cs336_basics.model.attention import scaled_dot_product_attention

torch.set_float32_matmul_precision('high')

def benchmark_attention(batch_size, d_model, seq_length, use_compile: bool = False):
    Q = torch.randn(batch_size, seq_length, d_model, device='cuda', requires_grad=True)
    K = torch.randn(batch_size, seq_length, d_model, device='cuda', requires_grad=True)
    V = torch.randn(batch_size, seq_length, d_model, device='cuda', requires_grad=True)

    attention_fn = scaled_dot_product_attention
    if use_compile:
        attention_fn = torch.compile(scaled_dot_product_attention, mode="max-autotune")

    # warmup 5次
    for _ in range(5):
        out = attention_fn(Q, K, V)
        loss = out.mean()
        loss.backward()
        Q.grad = K.grad = V.grad = None  # Clear gradients for the next iteration

    torch.cuda.synchronize()  # Ensure all CUDA operations are complete before timing

    # benchmark 100次
    forward_total_time = 0.0  
    backward_total_time = 0.0
    for _ in range(100):
        torch.cuda.synchronize() 
        # 计算前向时间 
        forward_start_time = time.perf_counter()
        out = attention_fn(Q, K, V)
        torch.cuda.synchronize()  
        forward_end_time = time.perf_counter()
        forward_total_time += (forward_end_time - forward_start_time)
        loss = out.mean()
        # 计算反向时间
        torch.cuda.synchronize()
        backward_start_time = time.perf_counter()
        loss.backward()
        Q.grad = K.grad = V.grad = None  # Clear gradients for the next iteration
        torch.cuda.synchronize()
        backward_end_time = time.perf_counter()
        backward_total_time += (backward_end_time - backward_start_time)
    # 打印100次前向传播和反向传播的时间
    print(f"Total forward time for 100 runs: {forward_total_time:.4f} seconds")
    print(f"Total backward time for 100 runs: {backward_total_time:.4f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for scaled_dot_product_attention")
    args = parser.parse_args()

    for d_model in [16]:
        for seq_length in [8192]:
            print(
                f"Benchmarking attention with d_model={d_model}, seq_length={seq_length}, compile={args.compile}"
            )
            try:
                benchmark_attention(8, d_model, seq_length, use_compile=args.compile)
            except RuntimeError as e:
                print(f"RuntimeError for d_model={d_model} and seq_length={seq_length}: {e}")
                exit(1)

if '__main__' == __name__:
    main()