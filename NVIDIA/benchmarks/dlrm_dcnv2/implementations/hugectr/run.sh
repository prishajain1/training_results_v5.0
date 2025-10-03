cd ~/training_results_v5.0/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr


source config_A3Mega_8x8.sh 

export DATADIR="/mnt/disks/persist/nvidia"

export LOGDIR="$(pwd)/logs_inference_128_524288"
mkdir -p $LOGDIR
echo "Logs and Nsys reports will be stored in: ${LOGDIR}"

export CONT="us-west1-docker.pkg.dev/tpu-prod-env-multipod/dlrm/dlrm_hugectr_a3mega_custom:latest"
export NEXP=1
export EV_SIZE=128

echo "Submitting Slurm job with EV_SIZE=${EV_SIZE}, BATCHSIZE=${BATCHSIZE}, MEMORY_CAP=${MEMORY_CAP_FOR_EMBEDDING}..."
sbatch \
  --nodes=${DGXNNODES} \
  --time=${WALLTIME} \
  --export=ALL \
  run_docker.sub
