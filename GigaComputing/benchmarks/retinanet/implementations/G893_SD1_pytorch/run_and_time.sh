#!/bin/bash

cd ../pytorch
export CONT=./nvdlfwea+mlperftv50+retinanet-amd+.sqsh
source config_G893-SD1_001x08x032.sh
export DATADIR=/path/to/datadir
export BACKBONE_DIR=/path/to/backbone_dir
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G893-SD1

sbatch -N $DGXNNODES -t $WALLTIME run.sub
