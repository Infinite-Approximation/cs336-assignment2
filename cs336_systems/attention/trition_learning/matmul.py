import torch
import triton
import triton.language as tl

@triton.jit
def naive_matmul_with_divisible_num(a_ptr, b_ptr, c_ptr, 
                 M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    不考虑L2缓存优化，并且都是可以M是BLOCK_SIZE_M的倍数，对于N，K同理
    """
    # 得到目前的program是处理哪个block
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 得到a矩阵的block中的元素对应的地址
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 得到b矩阵的block中的元素对应的地址
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # 开始计算
    # 用fp32进行累加
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    # 得到c矩阵的block中的元素对应的地址
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, c)

@triton.jit
def naive_matmul(a_ptr, b_ptr, c_ptr, 
                 M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    不考虑L2缓存优化，但是要考虑M不是BLOCK_SIZE_M的倍数，对于N，K同理
    """
    # 得到目前的program是处理哪个block
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # 处理M不是BLOCK_SIZE_M的整数倍的情况
    am_mask = offs_am[:, None] < M
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 处理N不是BLOCK_SIZE_N的整数倍的情况
    bn_mask = offs_bn[None, :] < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 得到a矩阵的block中的元素对应的地址
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 得到b矩阵的block中的元素对应的地址
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # 开始计算
    # 用fp32进行累加
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载a，b矩阵，并对越界地址进行掩码（填充0）
        ak_mask = offs_k[None, :] < K-k*BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=am_mask & ak_mask, other=0.0)
        bk_mask = offs_k[:, None] < K-k*BLOCK_SIZE_K
        b = tl.load(b_ptrs, mask=bk_mask & bn_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    # 得到c矩阵的block中的元素对应的地址
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    # 考虑M不是BLOCK_SIZE_M的倍数，N不是BLOCK_SIZE_N的倍数
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)   


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 64), )
    # naive_matmul_with_divisible_num[grid](a, b, c, M, N, K, 
    #                                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    #                                 32, 64, 64)
    naive_matmul[grid](a, b, c, M, N, K, 
                                    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                    32, 64, 64)
    return c

def main():
    torch.manual_seed(0)
    dtype = torch.float16
    a = torch.randn((257, 256), device='cuda:0', dtype=dtype)
    b = torch.randn((256, 256), device='cuda:0', dtype=dtype)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print("Triton output:" , triton_output)
    print("PyTorch output:" , torch_output)
    # 默认的rtol和atol的要求有时太严格了，triton的计算会和torch.matmul有一些差异，比如 Max diff: 0.03125
    # if torch.allclose(triton_output, torch_output):
    if torch.allclose(triton_output, torch_output, rtol=1e-2, atol=5e-2):
        print("结果正确！")
    else:
        print("结果错误！")
    print(f"Max diff: {(triton_output - torch_output).abs().max()}")

if __name__ == '__main__':
    main()