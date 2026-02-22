import argparse

import torch
import triton

from cs336_systems.attention.flash_attention import AttentionInPyTorch, FlashAttentionInTriton

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length for attention")
    argparser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for attention")
    argparser.add_argument("--precision", type=str, default="float32", help="Precision for attention computation (e.g., float16, float32)")
    args = argparser.parse_args()
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    # 创建随机输入
    Q = torch.randn((1, args.sequence_length, args.embedding_dim), dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn((1, args.sequence_length, args.embedding_dim), dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn((1, args.sequence_length, args.embedding_dim), dtype=dtype, device='cuda', requires_grad=True)
    dO = torch.randn((1, args.sequence_length, args.embedding_dim), dtype=dtype, device='cuda')

    attn_in_torch_forward = AttentionInPyTorch.apply
    flash_attn_in_triton_forward = FlashAttentionInTriton.apply

    def attention_in_pytorch():
        out = attn_in_torch_forward(Q, K, V, True)
        out.backward(dO)

    def attention_in_trition():
        out = flash_attn_in_triton_forward(Q, K, V, True)
        out.backward(dO)



    # 测试foward
    ms_torch_forward = triton.testing.do_bench(
        fn=lambda: attn_in_torch_forward(Q, K, V, True),
        warmup=25,
        rep=100,
    )

    ms_triton_forward = triton.testing.do_bench(
        fn=lambda: flash_attn_in_triton_forward(Q, K, V, True),
        warmup=25,
        rep=100,
    )

    print("=" * 60)
    print(f"Attention Forward in PyTorch: {ms_torch_forward:.2f} ms")
    print(f"Flash Attention Forward in Triton: {ms_triton_forward:.2f} ms")
    print(f"Speedup: {ms_torch_forward / ms_triton_forward:.2f}x")

    # 测试backward
    out_torch = attn_in_torch_forward(Q, K, V, True)
    ms_torch_backward = triton.testing.do_bench(
        fn=lambda: out_torch.backward(dO, retain_graph=True),
        warmup=25,
        rep=100,
    )

    out_triton = flash_attn_in_triton_forward(Q, K, V, True)
    dO = torch.randn_like(out_triton)
    ms_triton_backward = triton.testing.do_bench(
        fn=lambda: out_triton.backward(dO, retain_graph=True),
        warmup=25,
        rep=100,
    )

    print("=" * 60)
    print(f"Attention Backward in PyTorch: {ms_torch_backward:.2f} ms")
    print(f"Flash Attention Backward in Triton: {ms_triton_backward:.2f} ms")
    print(f"Speedup: {ms_torch_backward / ms_triton_backward:.2f}x")

    # 测试 end2end
    ms_torch_end2end = triton.testing.do_bench(
        fn=attention_in_pytorch,
        warmup=25,
        rep=100,
    )

    ms_triton_end2end = triton.testing.do_bench(
        fn=attention_in_trition,
        warmup=25, 
        rep=100,
    )

    print("=" * 60)
    print(f"Attention End2End in PyTorch: {ms_torch_end2end:.2f} ms")
    print(f"Flash Attention End2End in Triton: {ms_triton_end2end:.2f} ms")
    print(f"Speedup: {ms_torch_end2end / ms_triton_end2end:.2f}x")

if __name__ == '__main__':
    main()