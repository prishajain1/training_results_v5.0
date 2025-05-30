source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_blackwell.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=112
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1
export MODEL_SIZE="405b"

export TP_COMM_OVERLAP=True
export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True

export MAX_STEPS=500

# Binding
export BINDCMD="bindpcie --cpu=node"

export DGXNNODES=64
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=220
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
