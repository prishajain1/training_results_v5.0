#!/bin/bash

BASE_DIR=/mnt/localdisk/mlperf/lora
export DATADIR="${BASE_DIR}/data/gov_report"  # set your </path/to/dataset>
export MODEL="${BASE_DIR}/data/model"  # set your </path/to/dataset>
export LOGDIR="${BASE_DIR}/results"  # set the place where the output logs will be saved
export CONT=${BASE_DIR}/lora_v50_gb200.sqsh  # set the container url
export SLURM_MPI_TYPE=pmi2
export GLOO_SOCKET_IFNAME=eth0
#export CONT=nvcr.io/nvdlfwea/mlperftv50/llama2_70b_lora-amd:20250423  # set the container url
source config_GB200_1x4x4xtp1pp1cp2.sh  # select config and source it
#source config_GB200_2x4x1xtp1pp1cp1.sh  # select config and source it
#source config_GB200_18x4x1xtp1pp1cp8.sh  # select config and source it
sbatch --job-name=lora --exclusive --gpus-per-node=${DGXNGPU} -N $DGXNNODES -t $WALLTIME run.sub
#sbatch --exclusive --gpus-per-node=${DGXNGPU} -w instance[20250423002219] -N $DGXNNODES -t $WALLTIME run.sub
