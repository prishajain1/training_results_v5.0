#!/bin/bash
export SBATCH_NETWORK=sharp

## DL params
export BATCHSIZE=${BATCHSIZE:-10}
export NUMEPOCHS=${NUMEPOCHS:-6}
export LR=${LR:-0.0001} #0.000085
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
#export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --master-weights'}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --master-weights --eval-batch-size=32'}

## System run params
export DGXNNODES=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=20

export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))
export WALLTIME_RUNANDTIME=40

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

## Replace memset kernels with explicit zero-out kernels
export CUDNN_FORCE_KERNEL_INIT=1
