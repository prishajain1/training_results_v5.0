#!/bin/bash

BASE_DIR=/mnt/resource_nvme/mlperf/sd
export DATADIR=${BASE_DIR}/data/laion-400m/webdataset-moments-filtered-encoded
export COCODIR=${BASE_DIR}/data/coco2014
export CHECKPOINT_CLIP=${BASE_DIR}/chk_pnts/clip
export CHECKPOINT_FID=${BASE_DIR}/chk_pnts/inception
export CHECKPOINT_SD=${BASE_DIR}/chk_pnts/sd
export LOGDIR=${BASE_DIR}/logs  # set the place where the output logs will be saved
export CONT=${BASE_DIR}/sd_tv50.sqsh  # set the container url
export SLURM_MPI_TYPE=pmi2
source config_DGXB200_01x08x32.sh  # select config and source it
#source config_DGXB200_08x08x08.sh  # select config and source it
sbatch --exclusive --gpus-per-node=$DGXNGPU -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
