#!/bin/bash

set -e

ulimit -Sn 100000 || true

# export CUBLAS_LOGINFO_DBG=1
# export CUBLAS_LOGDEST_DBG=stderr
# export CUDA_LAUNCH_BLOCKING=1

export BATCHSIZE=48
export PACKING_FACTOR=1
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

: "${EVAL_ITER_SAMPLES:=100000}"
: "${NUM_EVAL_EXAMPLES:=10000}"
: "${EVAL_ITER_START_SAMPLES:=100000}"
: "${MAX_SAMPLES_TERMINATION:=14000000}"
: "${TARGET_MLM_ACCURACY:=0.720}"
: "${SUSTAINED_TRAINING_TIME:=0}" # Keep training if consuming less than SUSTAINED_TRAINING_TIME mins

#/workspace/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength/
torchrun \
   --nnodes=1 \
   --nproc-per-node=1 \
   --rdzv-backend=c10d \
   --rdzv-endpoint=127.0.0.1 \
   run_pretraining.py \
   --input_dir "/workspace/bert_data/packed_data/" \
   --output_dir "/workspace/bert_data/output/" \
   --do_train \
    --bert_config_path /workspace/bert_data/test/bert_config.json \
    --init_checkpoint /workspace/bert_data/test/model.ckpt-28252.pt \
    --dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha \
    --fused_dropout_add  --fused_gemm_gelu --packed_samples --use_transformer_engine2 \
    --cuda_graph_mode 'segmented' \
    --train_batch_size=${BATCHSIZE} \
    --learning_rate=${LR} \
    --opt_lamb_beta_1=${OPT_LAMB_BETA_1} \
    --opt_lamb_beta_2=${OPT_LAMB_BETA_2} \
    --warmup_proportion=${WARMUP_PROPORTION} \
    --warmup_steps=0.0 \
    --start_warmup_step=${START_WARMUP_STEP} \
    --max_steps=${MAX_STEPS} \
    --fp16 \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=${TARGET_MLM_ACCURACY} \
    --sustained_training_time=${SUSTAINED_TRAINING_TIME} \
    --weight_decay_rate=${WEIGHT_DECAY_RATE} \
    --max_samples_termination=${MAX_SAMPLES_TERMINATION} \
    --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES} --eval_iter_samples=${EVAL_ITER_SAMPLES} \
    --eval_batch_size=16 --eval_dir="/workspace/bert_data/hdf5/eval_varlength/" --num_eval_examples=${NUM_EVAL_EXAMPLES} \
    --output_dir=/workspace/bert/results/ \
    --distributed_lamb --dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-ag-pg=1 --dwu-num-blocks=1 \
    --gradient_accumulation_steps=${GRADIENT_STEPS}






