# 1 Assignment Overview
## 1.1 Profiling and Benchmarking
### 1.1.3 End-to-End Benchmarking
(a) 查看 `cs336_systems/benchmark/benchmark.py` 文件。
(b) 是在Batch_size=4, context_length=128的条件下运行的（去除了2.7B那个模型，因为显存不够）。

| 模型 | Forward Pass (秒) | Backward Pass (秒) |
|------|-------------------|--------------------|
| small | 0.0187 ± 0.0002 | 0.0423 ± 0.0015 |
| medium | 0.0430 ± 0.0012 | 0.0810 ± 0.0039 |
| large | 0.0570 ± 0.0012 | 0.1341 ± 0.0070 |
| xl | 0.0960 ± 0.0024 | 0.1812 ± 0.0037 |

一般来说backward pass(不包括optimizer.step)花费的时间是forward pass的两倍。

(c) 如果没有warming up的步骤，那么第一个前向传播会花费很多时间。
因为模型在第一次运行的时候会把对应的kernel加载到GPU中，以及对内存进行一个预分配之类的操作，这些都需要花时间，如果没有warmup，那么第一次运行的时间需要把这些都考虑进去。
而且没有warmup标准差很大很大。

| 模型 | Forward Pass (秒) | Backward Pass (秒) |
|------|-------------------|--------------------|
| small | 1.5542 ± 4.5447 | 0.1017 ± 0.2074 |
| medium | 2.8902 ± 8.5211 | 0.1578 ± 0.2629 |
| large | 4.2957 ± 12.6866 | 0.2249 ± 0.2539 |
| xl | 5.5856 ± 16.3981 | 0.3118 ± 0.0337 |

一步warmup的效果也不是很好，标准差依旧很大。 需要多步warmup才能使运行时间稳定下来。

| 模型 | Forward Pass (秒) | Backward Pass (秒) |
|------|-------------------|--------------------|
| small | 0.0377 ± 0.0197 | 0.0369 ± 0.0057 |
| medium | 0.0400 ± 0.0127 | 0.0878 ± 0.0006 |
| large | 0.0785 ± 0.0317 | 0.1114 ± 0.0185 |
| xl | 0.0984 ± 0.0477 | 0.2043 ± 0.0301 |

### 1.1.4 Nsight Systems Profiler
(a) 在Batch_size=4, context_length=128的条件下运行的

| 模型 | Forward Pass (秒) |
|------|-------------------|
| small | 0.0256 ± 0.0014 |
| medium | 0.0497 ± 0.0020 |
| large | 0.0729 ± 0.0063 |
| xl | 0.1079 ± 0.0023 |

NVTX 测量结果普遍比 timeit 高约 10-30%，这是因为 NVTX 包含了：
- nvtx.range_push/pop 本身的开销

(b) foward pass中最占用时间的kernel是 `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)`，占forward pass总时间的的49.2%

backward pass中最占用时间的kernel是 `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nt_align4>(T1::Params)`，占总时间的33.9%

forward和backward中最耗时的kernel不完全相同：
- Forward: `cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4` (tile size 128x128)
- Backward: `cutlass_80_tensorop_s1688gemm_256x128_16x3_nt_align4` (tile size 256x128)

两者都是Tensor Core GEMM kernel，但针对不同的矩阵形状使用了不同的优化配置。

(c) 
- `triton_poi_fused_mul_sigmoid_8` 占forward pass的4.0%，
- `triton_per_fused_add_div_mean_mul_pow_7` 占forward pass的2.2%

(d) 在一个完整的training step中， `vectorized_elementwise_kernel` 相关的操作占比最多，占用总时间55%，而基本都是optimizer.step()中在调用这些操作。所以optimizer.step()在整个training step中是十分耗时的。而矩阵乘法相关的操作占用的时间为总时间的35%。
在单独的forward pass中，矩阵乘法所花费的时间占比是88.4%，其他的占用都很小。

(e) 在单个自注意力计算中：

| 操作                             | 耗时             | 占比        |
| ------------------------------ | -------------- | --------- |
| **computing attention scores** | 552.600 µs     | **48.5%** |
| **computing softmax**          | 210.100 µs     | **18.4%** |
| **final matmul**               | 376.800 µs     | **33.1%** |

而它们的FLOPs分别是：

| 操作                         | FLOPs    | 占比                                  |
| -------------------------- | -------- | ----------------------------------- |
| computing attention scores | $2BL^2d_{model}$ | $\frac{2d_{model}}{4d_{model}+5H}$ (约 49.0%, d_{model}=768时) |
| computing softmax          | $5BL^2H$  | $\frac{5H}{4d_{model}+5H}$ (约 1.9%, d_{model}=768时)   |
| final matmul               | $2BL^2d_{model}$ | $\frac{2d_{model}}{4d_{model}+5H}$ (约 49.0%, d_{model}=768时) |

可以看到虽然softmax的FLOPs占比很低，只有1.9%，但是所耗费的时间却占比18.4%，说明对于softmax的优化不够，而矩阵乘法的优化很好。

