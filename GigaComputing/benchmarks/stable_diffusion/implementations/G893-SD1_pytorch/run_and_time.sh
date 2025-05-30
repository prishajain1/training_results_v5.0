#!/bin/bash

cd ../pytorch

source config_G893-SD1_01x08x32.sh
export CONT=./nvdlfwea+mlperftv50+stable_diffusion-amd+.sqsh
export DATADIR=/path/to/datadir
export COCODIR=/path/to/cocodir
export CHECKPOINT_CLIP=/path/to/checkpoint_clip
export CHECKPOINT_FID=/path/to/checkpoint_fid
export CHECKPOINT_SD=/path/to/checkpoint_sd
export LOGDIR=/path/to/logdir
export NEMOLOGS=/path/to/nvmologs
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G893-SD1

sbatch -N $DGXNNODES -t $WALLTIME run.sub
