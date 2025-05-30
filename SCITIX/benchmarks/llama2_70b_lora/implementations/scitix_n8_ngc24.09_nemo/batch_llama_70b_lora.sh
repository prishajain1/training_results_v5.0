#!/bin/bash

for i in {0..9}; do
    echo "Running iteration $i..."
    SEED=$((RANDOM + $$))
    mpirun --allow-run-as-root -np 64 --bind-to none -x MASTER_ADDR=172.21.76.112 \
    bash /data/mlperf_training/SCITIX/benchmarks/llama2_70b_lora/implementations/scitix_n8_ngc24.09_nemo_hpe/launch_scitix.sh $SEED \
    2>&1 | tee /data/mlperf_training/SCITIX/results/scitix_n8_ngc24.09_nemo_hpe/llama2_70b_lora/result_$i.txt
done

