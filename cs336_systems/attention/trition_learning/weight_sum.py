import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.autograd import gradcheck

@triton.jit
def weight_sum_fwd(x_ptr, weight_ptr, output_ptr,
                   x_stride_row, x_stride_dim,
                   weight_stride_dim,
                   output_stride_row,
                   ROWS, D,
                   ROW_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    output_block_ptr = tl.make_block_ptr(
		base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROW_TILE_SIZE,),
        block_shape=(ROW_TILE_SIZE,),
        order=(0,)
    )

    output = tl.zeros((ROW_TILE_SIZE,), dtype=tl.float64)

    for k in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero')
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weight_sum_backward(x_ptr, weight_ptr,
                        grad_output_ptr,
                        grad_x_ptr, partial_grad_weight_ptr, # 梯度矩阵
                        stride_gr, # grad_output跳到下一个元素需要的步数
                        stride_gxr, stride_gxd, # x的梯度矩阵的stride
                        stride_gwr, stride_gwd, # w的梯度矩阵(中间梯度矩阵)的stride
                        stride_wd, # weight的stride
                        stride_xr, stride_xd, # x的stride
                        NUM_ROWS, D,
                        ROW_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr):
    
    row_tile_idx = tl.program_id(axis=0)
    n_row_tiles = tl.num_programs(axis=0)

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROW_TILE_SIZE,),
        block_shape=(ROW_TILE_SIZE,),
        order=(0,)
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    # 需要将weight的梯度存放到一个中间梯度矩阵中，因为每个program计算出来的weight梯度不是完整的，
    # 还需要最后对这个中间梯度矩阵按列求和之后才可以
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr, 
        shape=(n_row_tiles, D),
        strides=(stride_gwr, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0)
    )

    for k in range(tl.cdiv(D, D_TILE_SIZE)):
        # 先计算x的梯度
        grad_output_block = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option='zero')
        weight_block = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero')
        # 利用外积得到 grad_x_block
        grad_x_block = grad_output_block[:, None] * weight_block[None, :]
        tl.store(grad_x_block_ptr, grad_x_block, boundary_check=(0, 1))

        # 再计算w的梯度
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        grad_weight_row = tl.sum(x_block * grad_output_block[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(0, 1))

        # 移动到下一个D_TILE
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # x.shape一般是 (B, S, D)
        D, output_dims = x.shape[-1], x.shape[:-1]
        ctx.input_shape = x.shape
        # 将batch size和序列长度展平，得到一个二维向量，n_rows是行数
        x = rearrange(x, "... d -> (...) d")
        # 保存输入以供反向传播使用
        ctx.save_for_backward(x, weight)
        # 创建结果矩阵
        n_rows = x.shape[0]
        output_flat = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        ctx.ROW_TILE_SIZE = 16
        grid = (triton.cdiv(n_rows, ctx.ROW_TILE_SIZE),)
        ctx.D_TILE_SIZE = triton.next_power_of_2(D)
        weight_sum_fwd[grid](x, weight, output_flat, 
                             x.stride(0), x.stride(1), 
                             weight.stride(0),
                             output_flat.stride(0),
                             ROWS=n_rows, D=D,
                             ROW_TILE_SIZE=ctx.ROW_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE)
        return output_flat.view(*output_dims)

    @staticmethod
    def backward(ctx, grad_out):
        grad_out_flat = grad_out.flatten()
        x, weight = ctx.saved_tensors
        n_rows, D = x.shape
        ROW_TILE_SIZE, D_TILE_SIZE = ctx.ROW_TILE_SIZE, ctx.D_TILE_SIZE
        n_row_tiles = triton.cdiv(n_rows, ROW_TILE_SIZE)
        # 创建grad_x来保存x的梯度
        grad_x = torch.empty_like(x)
        # 创建partial_grad_weight来保存每个program计算的weight梯度，最后需要按列求和
        partial_grad_weight = torch.empty((n_row_tiles, D), device=x.device, dtype=x.dtype)
        weight_sum_backward[(n_row_tiles, )](
            x, weight,
            grad_out_flat,
            grad_x, partial_grad_weight,
            grad_out_flat.stride(0), 
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            weight.stride(0),
            x.stride(0), x.stride(1),
            n_rows, D,
            ROW_TILE_SIZE=ROW_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE
        )
        return grad_x.view(ctx.input_shape), partial_grad_weight.sum(axis=0)

def test_weighted_sum_forward():
    """
    Test the forward pass of the weighted sum function.
    """
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D, device='cuda')
    weight = torch.randn(D, device='cuda')
    triton_output = WeightedSumFunc.apply(x, weight)
    torch_output = (x * weight).sum(dim=-1)
    assert torch.allclose(triton_output, torch_output), "The outputs do not match!"
    print("Test passed! The outputs match.")

def test_weighted_sum_backward():
    """
    Test the backward pass of the weighted sum function.
    """
    f_weightedsum = WeightedSumFunc.apply
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D, device='cuda', dtype=torch.double, requires_grad=True)
    weight = torch.randn(D, device='cuda',  dtype=torch.double, requires_grad=True)
    test = gradcheck(f_weightedsum, (x, weight), eps=1e-3, atol=1e-2)
    assert test, "Gradcheck failed!"
    print("Gradcheck passed! The gradients are correct.")

def main():
    test_weighted_sum_backward()

if __name__ == "__main__":
    main()