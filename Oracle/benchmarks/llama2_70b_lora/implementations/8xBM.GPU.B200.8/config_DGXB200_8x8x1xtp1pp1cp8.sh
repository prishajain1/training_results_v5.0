#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=700
export LR=0.0004
export MINIBS=1
export TP=1
export CP=8
export SP=0

export SBATCH_NETWORK=${SBATCH_NETWORK:-sharp}
export USE_SHARP=$( [[ ${SBATCH_NETWORK:-} == "sharp" ]] && echo True || echo False )

# Have to disable MCore CG and use our implementation, otherwise if using  with CP_EVAL, it fails with
# AssertionError: Tried replaying a cudagraph with different arguments than what if was created with!
export LAYER_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=1

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=8
export DGXNGPU=8
export WALLTIME_RUNANDTIME=23
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
