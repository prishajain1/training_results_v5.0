#!/bin/bash


source /mnt/vast/benchmark/perfalltoall/code/slurm_scripts/gpu_cluster/common/file_utils.sh
source bert_params.sh
source ../config_DGXH100_8x8x72x1_pack.sh

export PARTITION="POC-Zone1"
export NODES="hpf-s[0-7]"
export JOB_NAME="BERT_8_NODE_ZONE1"

# Enable for NVTX profiling
# export NVMLPERF_NSIGHT_LOCATION=/mnt/nfsshare/softwares/nsights/opt/nvidia/nsight-systems-cli/2025.1.1
# export NVTX_FLAG=1

create_output_directory "bert"
sbatch --output="$OUTPUT_DIR/bert-%j.out" -J $JOB_NAME -p $PARTITION -w $NODES ../run_with_network_metrics.sub
