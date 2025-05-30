source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_blackwell.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=32
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1
export MODEL_SIZE="405b"

export FP8_PARAM_GATHER=True

export TP_COMM_OVERLAP=True
export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True

export MAX_STEPS=450

# Binding
# export BINDCMD="bindpcie --cpu=node"

export DGXNNODES=624
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=180
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export TP_PP_DP_MAPPING=True
