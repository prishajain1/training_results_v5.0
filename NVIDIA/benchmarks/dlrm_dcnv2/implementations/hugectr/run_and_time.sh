#!/bin/bash
# Workaround: DO NOT include TCPXO paths
HUGECTR_LIB_PATH="/usr/local/hugectr/lib"
export LD_LIBRARY_PATH="${HUGECTR_LIB_PATH}:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu"

# Unset stock NCCL_VERSION to avoid confusion
unset NCCL_VERSION

set -eu
#set -ex

echo "--- Running ldconfig ---"
ldconfig
echo "--- ldconfig done ---"

# === DEBUGGING SECTION START ===
echo "--- [$(hostname)] run_and_time.sh DEBUG START (TCPXO Disabled) ---"
echo "SLURM_NODEID=${SLURM_NODEID:-UKN}, SLURM_PROCID=${SLURM_PROCID:-UKN}, SLURM_LOCALID=${SLURM_LOCALID:-UKN}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-UNSET}"

echo "--- Running ldconfig -p | grep libnccl ---"
ldconfig -p | grep libnccl

HUGECTR_LIB="${HUGECTR_LIB_PATH}/libhuge_ctr_shared.so"
echo "--- Checking for ${HUGECTR_LIB} ---"
if [ -f "${HUGECTR_LIB}" ]; then
    ls -l "${HUGECTR_LIB}"
    echo "--- LDD on ${HUGECTR_LIB} for NCCL ---"
    ldd "${HUGECTR_LIB}" | grep libnccl || echo "${HUGECTR_LIB} does not directly link libnccl"
else
    echo "ERROR: ${HUGECTR_LIB} not found" >&2
    exit 2
fi

echo "--- Key NCCL Env Vars ---"
env | grep NCCL_ | sort
echo "--- [$(hostname)] run_and_time.sh DEBUG END ---"

# default value for DLRM_BIND only if it is not already defined
: ${DLRM_BIND:=}

ARGS=""
[ -n "${MODE:-}" ] && ARGS+=" --mode ${MODE}"
[ -n "${WARMUP_STEPS:-}" ] && ARGS+=" --warmup_steps ${WARMUP_STEPS}"
[ -n "${BENCHMARK_STEPS:-}" ] && ARGS+=" --benchmark_steps ${BENCHMARK_STEPS}"
[ -n "${OPTIMIZER:-}" ] && ARGS+=" --optimizer ${OPTIMIZER}"
[ -n "${BATCHSIZE:-}" ] && ARGS+=" --batchsize ${BATCHSIZE}"
[ -n "${BATCHSIZE_EVAL:-}" ] && ARGS+=" --batchsize_eval ${BATCHSIZE_EVAL}"
[ -n "${LEARNING_RATE:-}" ] && ARGS+=" --lr ${LEARNING_RATE}"
[ -n "${MEMORY_CAP_FOR_EMBEDDING:-}" ] && ARGS+=" --memory_cap_for_embedding ${MEMORY_CAP_FOR_EMBEDDING}"
[ -n "${DECAY_START:-}" ] && ARGS+=" --decay_start ${DECAY_START}"
[ -n "${DECAY_STEPS:-}" ] && ARGS+=" --decay_steps ${DECAY_STEPS}"
[ "${ENABLE_TF32_COMPUTE:-false}" = true ] && ARGS+=" --enable_tf32_compute"
[ "${USE_MIXED_PRECISION:-false}" = true ] && ARGS+=" --use_mixed_precision"
[ -n "${SCALER:-}" ] && ARGS+=" --scaler ${SCALER}"
[ "${GEN_LOSS_SUMMARY:-false}" = true ] && ARGS+=" --gen_loss_summary"
[ "${USE_ALGORITHM_SEARCH:-true}" = false ] && ARGS+=" --disable_algorithm_search"
[ -n "${SHARDING_PLAN:-}" ] && ARGS+=" --sharding_plan ${SHARDING_PLAN}"
[ -n "${DP_SHARDING_THRESHOLD:-}" ] && ARGS+=" --dp_sharding_threshold ${DP_SHARDING_THRESHOLD}"
[ -n "${MAX_ITER:-}" ] && ARGS+=" --max_iter ${MAX_ITER}"
[ -n "${DISPLAY_INTERVAL:-}" ] && ARGS+=" --display_interval ${DISPLAY_INTERVAL}"
[ -n "${EVAL_INTERVAL:-}" ] && ARGS+=" --eval_interval ${EVAL_INTERVAL}"
[ -n "${MAX_EVAL_BATCHES:-}" ] && ARGS+=" --max_eval_batches ${MAX_EVAL_BATCHES}"
[ -n "${AUC_THRESHOLD:-}" ] && ARGS+=" --auc_threshold ${AUC_THRESHOLD}"
[ -n "${DGXNGPU:-}" ] && ARGS+=" --num_gpus_per_node ${DGXNGPU}"
[ -n "${MEM_COMM_BW_RATIO:-}" ] && ARGS+=" --mem_comm_bw_ratio ${MEM_COMM_BW_RATIO}"
[ -n "${SEED:-}" ] && ARGS+=" --seed ${SEED}"
[ -n "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-}" ] && ARGS+=" --minimum_training_time ${MINIMUM_TRAINING_TIME:-0}"
[ -n "${TRAIN_DATA:-}" ] && ARGS+=" --train_data ${TRAIN_DATA}"
[ -n "${VAL_DATA:-}" ] && ARGS+=" --val_data ${VAL_DATA}"

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${SLURM_LOCALID:-0}"

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
fi

: "${LOGGER:=""}"
echo "DLRM_BIND is set to \"${DLRM_BIND}\""
echo "LOGGER is set to \"${LOGGER}\""

NSYS_OUT="/logs/dlrm_nsys_rank${SLURM_PROCID}_$(date +%Y%m%d-%H%M%S)"
PROFILE_CMD="${LOGGER} ${DLRM_BIND} python3 /workspace/train.py ${ARGS}"
echo "Node ${node_rank} Rank ${SLURM_PROCID}: Starting profiling/training..."
echo "PROFILE_CMD: ${PROFILE_CMD}"

nsys profile -w true -t cuda,nvtx,cudnn,cublas -o ${NSYS_OUT} --force-overwrite true \
  bash -c "${PROFILE_CMD}"

ret_code=${PIPESTATUS[0]}
if [[ $ret_code != 0 ]]; then
  echo "Node ${node_rank} Rank ${SLURM_PROCID}: Nsys profiling or training command FAILED with code ${ret_code}" >&2
fi
echo "Node ${node_rank} Rank ${SLURM_PROCID}: Nsys profiling finished. Report: ${NSYS_OUT}.qdrep"
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
    result=$(( $end - $start ))
    echo "RESULT,DLRM,,$result,nvidia,$start_fmt"
fi