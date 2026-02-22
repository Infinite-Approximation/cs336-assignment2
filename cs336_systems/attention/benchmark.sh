seq_len=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
embed_dim=(16 32 64 128)
precision=(bfloat16 float32)

for seq in ${seq_len[@]}; do
    for dim in ${embed_dim[@]}; do
        for prec in ${precision[@]}; do
            echo "Running benchmark with sequence length: $seq, embedding dimension: $dim, precision: $prec"
            uv run python cs336_systems/attention/benchmark.py --sequence_length $seq --embedding_dim $dim --precision $prec
        done
    done
done