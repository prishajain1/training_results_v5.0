#!/bin/bash

export JAX_TRACEBACK_FILTERING=off
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0

# checkpoint interval for num of pics consumed
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-512000}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-10}
NUM_DEVICES=${NUM_DEVICES:-64}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-5000}
LR=${LR:-1.75e-4}
WARM_UP=${WARM_UP:-0.1}
METRICS_INTERVAL=${METRICS_INTERVAL:-100}
# different random seed for each run
SEED=${SEED:-0}

EVAL_OUT_DIR=/tmp/outputs
mkdir -p $EVAL_OUT_DIR
# training
RUN_NAME=${RUN_NAME:-"mlperf_e2e"}
OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY:-gs://${your_bucket_name}/$USER/$RUN_NAME}
DATA_DIR=gs://${your_bucket_name}/laion400m/raw_data/tf_records_512_encoder_state_fp32
bucket_name=${your_bucket_name}

python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name="$RUN_NAME" base_output_directory="$OUTPUT_DIRECTORY" train_data_dir="${DATA_DIR}" \
per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" split_head_dim=True  attention=flash  norm_num_groups=16 \
eval_at_checkpoint=False \
train_new_unet=True \
warmup_steps_fraction="${WARM_UP}" learning_rate="${LR}" \
noise_offset=-1.0 input_peturbation=-1.0 prediction_type='v_prediction' snr_gamma=-1.0 \
learning_rate_scheduler='linear' \
timestep_spacing='trailing' \
upload_images=False \
caption_coco_file="gs://${bucket_name}/cocodata/val2014_30k_padded.tsv" \
images_directory="$EVAL_OUT_DIR/" \
stat_output_directory="output/" \
stat_output_file="output/stats.npz" \
stat_coco_file="gs://${bucket_name}/cocodata/val2014_30k_stats.npz" \
clip_cache_dir="clip_cache_dir" \
seed="$SEED" \
checkpoint_every="$CHECKPOINT_EVERY" max_train_steps="$MAX_TRAIN_STEPS" metrics_period="$METRICS_INTERVAL" 2>&1 | tee /tmp/log

sleep 60


if [[ $(grep "MLLOG" /tmp/log | wc -l) -gt 0 ]];then
  python src/maxdiffusion/report_end.py --metrics-path="${OUTPUT_DIRECTORY}"/"${RUN_NAME}"/eval_metrics.csv --mllog-path=/tmp/log 2>&1 | tee -a /tmp/log
  gsutil cp /tmp/log "${OUTPUT_DIRECTORY}"/"${RUN_NAME}"/log_"${MEGASCALE_SLICE_ID}"_"${TPU_WORKER_ID}"
fi
