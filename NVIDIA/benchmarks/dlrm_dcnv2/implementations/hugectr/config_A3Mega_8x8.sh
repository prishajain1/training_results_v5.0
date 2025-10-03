#!/bin/bash
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## Mode
export MODE="benchmark_inference" # Or "full_train" if you switch back

## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=524288            
export BATCHSIZE_EVAL=524288       
export WARMUP_STEPS=50
export BENCHMARK_STEPS=200
export USE_MIXED_PRECISION=true
export SCALER=16384

export MEMORY_CAP_FOR_EMBEDDING=50 
export TRAIN_DATA="/data/train_data.bin"
export VAL_DATA="/data/val_data.bin"

## System run parms
export DGXNNODES=16
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=15 # Adjusted for smaller batch

export WALLTIME=$((15 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
