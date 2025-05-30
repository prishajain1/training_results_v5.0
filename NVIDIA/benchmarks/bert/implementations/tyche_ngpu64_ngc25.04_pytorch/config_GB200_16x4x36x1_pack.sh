## DL params                                                                                                                                                                                                          
export BATCHSIZE=36
export GRADIENT_STEPS=1
export PACKING_FACTOR=2
#export LR=0.0029293
export LR=0.00258
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=780
export OPT_LAMB_BETA_1=0.7206
export OPT_LAMB_BETA_2=0.78921
export START_WARMUP_STEP=-200000
export WARMUP_STEPS=170200
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=1024.0

export USE_FLASH_ATTENTION=1
#export SBATCH_NETWORK=sharp
export EXTRA_PARAMS=" --dwu-group-size=4  --dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu  --packed_samples --use_transformer_engine2 --cuda_graph_mode 'segmented' --use_cuda_graph --eval_cuda_graph "
export PHASE=2
export EVAL_ITER_START_SAMPLES=200000
export EVAL_ITER_SAMPLES=200000

## System run parms 
export DGXNGPU=4                                                                                                                                                                                                  
export DGXNNODES=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=3
if [[ "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-0}" == "1" ]]; then
  export WALLTIME_RUNANDTIME=$((${WALLTIME_RUNANDTIME} + 15))
  export SUSTAINED_TRAINING_TIME=11
fi
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]] || [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export WALLTIME_RUNANDTIME=$((${WALLTIME_RUNANDTIME} + 5))
  ## gpc frequency at maxQ and minEDP point
  export MAXQ_CLK=1515
  export MINEDP_CLK=1650
fi
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

## System config params                                                                                                                                                                                               
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

export USE_DDP=1

export DATADIR_PHASE2=${DATADIR_PHASE2_PACKED}
