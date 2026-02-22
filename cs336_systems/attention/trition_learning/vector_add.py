import torch
import triton
import triton.language as tl

@triton.jit
def vector_add(x_ptr, y_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x_block = tl.load(x_ptr + offsets, mask=mask)
    y_block = tl.load(y_ptr + offsets, mask=mask)
    res = x_block + y_block
    tl.store(o_ptr + offsets, value=res, mask=mask)

def main():
    n = 1_000_000
    x = torch.randn(n, device='cuda:0')
    y = torch.randn(n, device='cuda:0')
    o = torch.empty_like(x)
    BLOCK_SIZE = 1024
    # grid = (triton.cdiv(n, BLOCK), )
    # 使用meta来根据tl.constexpr变量自动计算
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    vector_add[grid](x, y, o, n, BLOCK_SIZE)
    if torch.allclose(o, x + y):
        print("结果正确！")
    else:
        print("结果错误！")

if __name__ == '__main__':
    main()