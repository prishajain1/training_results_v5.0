#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
#source $(dirname ${BASH_SOURCE[0]})/config_XD670_common.sh
export SUBMISSION_ORG=HPE
export MLPERF_SUBMITTER=HPE
export MLPERF_SYSTEM_NAME="HPE Cray XD670"
export MLPERF_STATUS="Available onprem"
export MLPERF_DIVISION=closed
export MLPERF_NUM_NODES=4

# hyperparameters
export MAX_STEPS=1024
export LR=0.0005
export MINIBS=2
export TP=2
export CP=1
export SKIP_EVALS=3
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export VBOOST_VALUE=1
export FP8_ACTIVATION=True 
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export TP_COMM_OVERLAP=1
export NCCL_MIN_CTAS=32
export UCX_TLS=self,tcp

# system parameters
#export DGXNNODES=2
#export WALLTIME_RUNANDTIME=50
#export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))


############### H100 x4 node config:
# hyperparameters
#export MINIBS=4
#export LR=0.00036

#export TP=4
#export PP=1
#export CP=2
#export SP=1
#export TP_COMM_OVERLAP=True

#export FP8=True
#export FP8_AMAX_ALGO=max
#export FP8_REDUCE_AMAX=True
#export FP8_AMAX_HISTORY=32

#export SKIP_EVALS=3
#export HYDRA_FULL_ERROR=1
#export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=4
#export SBATCH_NETWORK=sharp
#export SHARP=True
#export NCCL_SHARP_DISABLE=1

#export WALLTIME_MINUTES=15
#export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
export WALLTIME_RUNANDTIME=15
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

