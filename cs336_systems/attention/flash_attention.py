import torch
import triton
import triton.language as tl
from einops import einsum

torch.set_float32_matmul_precision('high')

class AttentionInPyTorch(torch.autograd.Function):
    """
    使用Pytorch实现Attention，这是最原始的版本
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        d_k = Q.shape[-1]
        S = einsum(Q, K, "... i d_k, ... j d_k -> ... i j") / d_k**0.5
        if is_causal:
            nq, nk = Q.shape[-2], K.shape[-2]
            q_idx = torch.arange(nq, device=Q.device)
            k_idx = torch.arange(nk, device=K.device)
            casual_mask = q_idx[:, None] < k_idx[None, :]
            S = S.masked_fill(casual_mask, -1e6)
            
        P = torch.softmax(S, dim=-1)
        O = einsum(P, V, "... i j, ... j d_k -> ... i d_k")
        ctx.save_for_backward(Q, K, V, P)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, P = ctx.saved_tensors
        d_k = Q.shape[-1]
        scale = 1 / d_k**0.5

        dV = einsum(P, grad_out, "... i j, ... i d_k -> ... j d_k")
        dP = einsum(grad_out, V, "... i d_k, ... j d_k -> ... i j")
        # 计算dS
        D = torch.sum(P * dP, dim=-1, keepdim=True)
        dS = P * (dP - D)
        dQ = einsum(dS, K, "... i j, ... j d_k -> ... i d_k") * scale
        dK = einsum(dS, Q, "... i j, ... i d_k -> ... j d_k") * scale

        return dQ, dK, dV, None


class FlashAttentionInPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        d_k = Q.shape[-1]
        S = einsum(Q, K, "... i d_k, ... j d_k -> ... i j") / d_k**0.5
        if is_causal:
            nq, nk = Q.shape[-2], K.shape[-2]
            q_idx = torch.arange(nq, device=Q.device)
            k_idx = torch.arange(nk, device=K.device)
            casual_mask = q_idx[:, None] < k_idx[None, :]
            S = S.masked_fill(casual_mask, -1e6)
            
        L = torch.logsumexp(S, dim=-1)
        P = torch.softmax(S, dim=-1)
        O = einsum(P, V, "... i j, ... j d_k -> ... i d_k")
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_kernel_in_pytorch(Q, K, V, O, grad_out, L)
        return dQ, dK, dV, None

@triton.autotune(
    configs=[
        triton.Config({"Q_TILE_SIZE": 16, "K_TILE_SIZE": 16}, num_warps=4, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_warps=4, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 32}, num_warps=8, num_stages=3),
    ],
    key=["N_QUERIES", "N_KEYS", "D", "is_causal"]
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # program indices
    query_tile_idx = tl.program_id(axis=0)
    batch_idx = tl.program_id(axis=1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_block = tl.load(Q_block_ptr).to(tl.float32)
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    # 每行的最大值
    m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    # softmax的分母
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    # 输出O
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(0, T_k):
        K_block = tl.load(K_block_ptr).to(tl.float32)
        V_block = tl.load(V_block_ptr).to(tl.float32)
        # Compute tile of pre-softmax attention scores
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale
        if is_causal:
            # 进行掩码 
            q_idx = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] >= k_idx[None, :]
            S_block = tl.where(mask, S_block, -1e6)

        # 更新每行的最大值
        row_max = tl.max(S_block, axis=1)
        # 必须使用m_prev，后面需要使用m_prev和m(也就是m_new)来计算因子
        m_prev = m
        m = tl.maximum(m_prev, row_max)
        # 计算 tiled_P，也就是 unnormalized softmax values
        tiled_P = tl.exp(S_block - m[:, None])
        # 更新softmax分母
        l = tl.exp(m_prev - m) * l + tl.sum(tiled_P, axis=1)
        # 更新O
        O = tl.exp(m_prev - m)[:, None] * O + tl.dot(tiled_P.to(V_block.dtype), V_block)

        # 移动 K_block_ptr 和 V_block_ptr
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O = O / l[:, None]
    L = m + tl.log(l)
    # 保存
    O_out = O.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_out)
    tl.store(L_block_ptr, L)



def flash_backward_kernel_in_pytorch(Q, K, V, O, dO, L, is_causal=False):
    """
    利用pytorch而不是triton实现flash attention的backward
    """
    # 反向传播之前先计算D
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    d_k = Q.shape[-1]
    # 计算未归一化的attention score
    scale = 1 / d_k**0.5
    S = einsum(Q, K, "... i d_k, ... j d_k -> ... i j") * scale
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        q_idx = torch.arange(nq, device=Q.device)
        k_idx = torch.arange(nk, device=K.device)
        casual_mask = q_idx[:, None] < k_idx[None, :]
        S = S.masked_fill(casual_mask, -1e6)

    # 借助 L 计算归一化后的attention score
    P = torch.exp(S - L.unsqueeze(-1))
    # 计算dV
    dV = einsum(P, dO, "... i j, ... i d_k -> ... j d_k")
    # 计算dP
    dP = einsum(dO, V, "... i d_k, ... j d_k -> ... i j")
    # 计算dS
    dS = P * (dP - D)
    # 计算dQ
    dQ = einsum(dS, K, "... i j, ... j d_k -> ... i d_k") * scale
    # 计算dK
    dK = einsum(dS, Q, "... i j, ... i d_k -> ... j d_k") * scale
    return dQ, dK, dV

# 使用torch.compile进行编译
flash_backward_kernel_in_pytorch = torch.compile(flash_backward_kernel_in_pytorch)

class FlashAttentionInTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_QUERYS, D = Q.shape
        B, N_KEYS, D = K.shape
        scale = 1 / D**0.5
        B_q, B_k = 32, 32
        T_q, T_k = triton.cdiv(N_QUERYS, B_q), triton.cdiv(N_KEYS, B_k)
        O = torch.empty((B, N_QUERYS, D), dtype=Q.dtype, device=Q.device)
        L = torch.empty((B, N_QUERYS), dtype=torch.float32, device=Q.device)
        grid = lambda META: (triton.cdiv(N_QUERYS, META['Q_TILE_SIZE']), B)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERYS, N_KEYS,
            scale,
            D=D,
            # 下面两个参数通过autotune自动传入
            # Q_TILE_SIZE=B_q,
            # K_TILE_SIZE=B_k,
            is_causal=is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_kernel_in_pytorch(Q, K, V, O, grad_out, L, ctx.is_causal)
        return dQ, dK, dV, None


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    # sequence_length = 16384
    sequence_length = 4096
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(FlashAttentionInTriton.apply)
 
    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    # results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    results = triton.testing.do_bench(flash_forward_backward, rep=100, warmup=100)
    print(f"Flash Attention forward + backward: {results:.2f} ms")

if __name__ == "__main__":
    test_timing_flash_forward_backward()