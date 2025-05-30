source $(dirname ${BASH_SOURCE[0]})/config_common.sh

export MINIBS=288
export TENSOR_MODEL_PARALLEL=8
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export TP_COMM_OVERLAP=True

export DGXNNODES=64
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=420
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export MODEL_SIZE="405b"

export MICRO_BATCH_SIZE=1

export MAX_STEPS=300

export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True
