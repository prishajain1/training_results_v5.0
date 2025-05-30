## DL params
export BATCHSIZE=48
export PACKING_FACTOR=2
export GRADIENT_STEPS=1
export LR=0.00096
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=3680
export OPT_LAMB_BETA_1=0.60466
export OPT_LAMB_BETA_2=0.85437
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=1024.0

export EXTRA_PARAMS="--dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu --packed_samples --use_transformer_engine2 --cuda_graph_mode 'segmented' --use_cuda_graph "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=8

if [[ "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-0}" == "1" ]]; then
  export WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 15))  
  export SUSTAINED_TRAINING_TIME=11
fi
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]] || [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 10))
  ## gpc frequency at maxQ and minEDP point
  export MAXQ_CLK=1305
  export MINEDP_CLK=1650
fi
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh

# export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1

# ## force to use packed trainset
# export DATADIR_PHASE2=${DATADIR_PHASE2_PACKED}

export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=ALL
export UCX_NET_DEVICES="gpu0_eth,gpu1_eth,gpu2_eth,gpu3_eth,gpu4_eth,gpu5_eth,gpu6_eth,gpu7_eth"
export NCCL_IB_HCA="gpu0_rdma,gpu1_rdma,gpu2_rdma,gpu3_rdma,gpu4_rdma,gpu5_rdma,gpu6_rdma,gpu7_rdma"
export NCCL_SOCKET_IFNAME="gpu0_eth,gpu1_eth,gpu2_eth,gpu3_eth,gpu4_eth,gpu5_eth,gpu6_eth,gpu7_eth"
export NCCL_IB_GID_INDEX=3
export NCCL_PXN_DISABLE=0
export NCCL_TEST=0
export JOB_NAME="BERT_2_NODE"

export BASEDIR="/mnt/vast/images/bert"
export CONT="${BASEDIR}/bert_latest+20241129_dev2.sqsh"
export LOGDIR="/mnt/vast/network_tests_bert/testing/"

export DATAPATH="/home/hpf/mlperf_data/bert"
export JOB_NAME="BERT_2_NODE"

export DATADIR="${DATAPATH}/packed_data"
export EVALDIR="${DATAPATH}/hdf5/eval_varlength/"
export DATADIR_PHASE2="${DATADIR}"
export CHECKPOINTDIR_PHASE1="${DATAPATH}/phase1/"
export CHECKPOINTDIR="${CHECKPOINTDIR_PHASE1}"

export MLPERF_SYSTEM_NAME="HPF6"
export MLPERF_CLUSTER_NAME="HPF_CISCO"
export SLURM_MPI_TYPE="pmix"

export NEXP=1
export PARTITION="POC-Zone2"
export OUTPUT_DIR=/mnt/vast/slurm_outputs
export NODES="hpf-s[14-15]"