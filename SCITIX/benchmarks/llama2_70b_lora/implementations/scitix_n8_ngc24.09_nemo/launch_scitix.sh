#!/usr/bin/env bash

# Copyright (c) 2024, Scitix Tech PTE. LTD. All rights reserved.
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

TEST_DIR=$(cd $(dirname $0) && pwd)
export SEED=$1
export CUDA_DEVICE_MAX_CONNECTION=1
export ENABLE_RERUNS=1
export CREATE_TENSORBOARD_LOGGER=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIENT_ACCUMULATION_STEPS=2
source $TEST_DIR/config_H100_8x8x1xtp4pp1cp2.sh
bash $TEST_DIR/mpirun2pytorch $TEST_DIR/run_and_time_scitix.sh
