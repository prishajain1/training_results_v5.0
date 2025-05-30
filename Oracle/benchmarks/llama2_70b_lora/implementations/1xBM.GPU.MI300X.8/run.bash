export DATADIR=/mlperf/lora/amd/data/
export LOGDIR=/mlperf/lora/amd/results
export CONT=rocm/amd-mlperf:llama2_70b_training_5.0         
source config_MI300X_1x8x1.sh  # use appropriate config
export NEXP=10
./run_with_docker.sh
