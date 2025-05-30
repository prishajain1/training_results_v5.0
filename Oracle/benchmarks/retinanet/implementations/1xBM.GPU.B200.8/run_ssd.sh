#!/bin/bash

BASE_DIR=/mnt/resource_nvme/mlperf/ssd
export DATADIR=${BASE_DIR}/data/open-images-v6
export BACKBONE_DIR=${BASE_DIR}/chk_pnts
export LOGDIR=${BASE_DIR}/logs
export CONT=${BASE_DIR}/ssd_tv50.sqsh  # set the container url
export SLURM_MPI_TYPE=pmi2
source config_DGXB200_001x08x032.sh  # select config and source it
sbatch -w gpu-153 --exclusive --gpus-per-node=8  -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
