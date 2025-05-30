#!/bin/bash

cd ../pytorch
export CONT=./nvdlfwea+mlperftv50+bert-amd+.sqsh
source config_G893-SD1_1x8x48x1_pack.sh
export EVALDIR=/path/to/evaldir
export DATADIR_PHASE2=/path/to/datadir_phase2
export CHECKPOINTDIR_PHASE1=/path/to/checkpointdir_phase1
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G893-SD1

sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
