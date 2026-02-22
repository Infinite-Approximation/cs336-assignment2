backends=(gloo nccl)
data_sizes=(1 10 100 1024) # MB
for backend in ${backends[@]}; do
    for data_size in ${data_sizes[@]}; do
        uv run python cs336_systems/collective_communication/benchmark_dist_app.py \
            --backend $backend \
            --world_size 2 \
            --data_size $data_size
    done
done