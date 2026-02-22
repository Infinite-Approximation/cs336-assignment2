import argparse
import os
import torch
import time
import torch.multiprocessing as mp
import torch.distributed as dist

def set_up(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def run_benchmark(rank, backend, world_size, data_size):
    set_up(rank, world_size, backend)
    # Create a tensor of the specified size
    # data_type = float32, 所以一共有 data_size/4 个数
    data_size_in_bytes = data_size * 1024 * 1024
    num_data = data_size_in_bytes // 4
    device = f"cuda:{rank}" if backend == 'nccl' else "cpu"
    data = torch.rand(num_data, dtype=torch.float32, device=device)
    
    # warm up 
    for _ in range(5):
        dist.all_reduce(data, async_op=False) # 如果后端是nccl，会等nccl通信完成，而不是gpu完成同步，所以需要torch.cuda.synchronize()来确保gpu完成同步
        if backend == 'nccl':
            torch.cuda.synchronize()

    # 开始计时
    if backend == 'nccl':
        torch.cuda.synchronize()
    start = time.time()
    dist.all_reduce(data, async_op=False)
    if backend == 'nccl':
        torch.cuda.synchronize()
    end = time.time()

    local_time = torch.tensor([end-start], device=device, dtype=torch.float32)
    gathered_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_times, local_time)
    # 现在每个 rank 都有了所有 rank 的时间
    if rank == 0:
        times_ms = [t.item() * 1000 for t in gathered_times]
        print(f"Word size: {world_size}, Backend: {backend}, Data_size: {data_size}(MB), Average Time: {sum(times_ms)/len(times_ms):.2f}ms")

    dist.barrier()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Benchmark distributed communication')
    parser.add_argument('--backend', type=str, default='gloo', help='The backend to use for distributed communication')
    parser.add_argument('--world_size', type=int, default=2, help='Number of processes to use for distributed communication')
    parser.add_argument('--data_size', type=int, default=1, help='Size of the data to be communicated in MB')
    args = parser.parse_args()
    mp.spawn(run_benchmark, args=(args.backend, args.world_size, args.data_size), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    main()

