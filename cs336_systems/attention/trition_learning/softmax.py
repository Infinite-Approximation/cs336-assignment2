import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)  # 是第几个program，就从第几行开始处理
    row_step = tl.num_programs(0)  # 跳row_step步到下一个行
    for row_idx in tl.range(
        row_start, n_rows, row_step, num_stages=num_stages
    ):  # 利用流水线来增加效率
        row_start_ptr = row_idx * input_row_stride + input_ptr
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        # 加载第 row_idx 行的数据
        row_data = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # 进行计算
        row_data_minus_max = row_data - tl.max(row_data, axis=0)
        numerator = tl.exp(row_data_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # 写回到output
        output_start_ptr = row_idx * output_row_stride + output_ptr
        output_ptrs = output_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def simple_call_softmax_kernel(x):
    """
    一行对应一个线程块
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    y = torch.empty_like(x)
    softmax_kernel[grid](
        x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def softmax(x):
    """
    根据设备属性来找到运行softmax_kernel的最优方法
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 获取设备信息
    properties = driver.active.utils.get_device_properties(0)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    # 设定流水线阶段数量
    num_stages = 4 if SIZE_SMEM > 200_000 else 2
    # 设置num_warps，增加一个thread block中的线程数
    num_warps = 8
    y = torch.empty_like(x)
    # 通过warmup来计算occupacy，只使用一个thread block，触发编译
    kernel = softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    # print("=== Kernel 对象信息 ===")
    # print(f"kernel 类型: {type(kernel)}")
    # print(f"kernel.n_regs: {kernel.n_regs}")  # 每个线程使用的寄存器数量
    # print(f"kernel.metadata: {kernel.metadata}")
    # print(f"kernel.metadata.shared: {kernel.metadata.shared}")  # 共享内存大小
    # print(f"kernel.metadata.num_warps: {kernel.metadata.num_warps}")  # warp 数量
    # print(f"kernel.metadata.num_stages: {kernel.metadata.num_stages}")  # 流水线阶段
    # print(f"kernel.metadata.num_ctas: {kernel.metadata.num_ctas}")  # CTA 数量
    # print("========================")
    # 得到每个线程使用了多少寄存器。这个在编译kernel的时候就决定了要给每个block分配的资源
    n_regs = kernel.n_regs
    # 每个block用了多少共享内存
    size_smem = kernel.metadata.shared
    # 计算occupacy
    occupacy = NUM_REGS // (WARP_SIZE * num_warps * n_regs)
    occupacy = min(occupacy, SIZE_SMEM // size_smem)
    # 计算需要多少个block
    num_programs = NUM_SM * occupacy
    num_programs = min(num_programs, n_rows)
    softmax_kernel[(num_programs,)](
        x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages
    )
    return y


def main():
    n_rows, n_cols = 1024, 1024
    x = torch.randn(n_rows, n_cols, device="cuda:0")
    # y_triton = simple_call_softmax_kernel(x)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    if torch.allclose(y_triton, y_torch):
        print("结果正确！")
    else:
        print("结果错误！")


if __name__ == "__main__":
    main()
