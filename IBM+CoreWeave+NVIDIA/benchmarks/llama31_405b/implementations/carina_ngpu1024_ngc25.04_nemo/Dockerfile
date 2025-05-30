# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${FROM_IMAGE_NAME}

# Document build setup
ARG FROM_IMAGE_NAME
ENV CUSTOM_FROM_IMAGE_NAME ${FROM_IMAGE_NAME}

# Custom libraries version
WORKDIR /workspace/

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

## 0. Pytorch Checkpoint size patch
WORKDIR /workspace/pytorch
COPY ./pytorch_ckpt.patch /workspace/pytorch/pytorch_ckpt.patch
# torch path obtained by:
# python3 -c "import torch; print(torch.__file__.replace('torch/__init__.py', ''))"
RUN patch --directory=/usr/local/lib/python3.12/dist-packages -p1 < /workspace/pytorch/pytorch_ckpt.patch
WORKDIR /workspace/

RUN pip install numcodecs==0.13.1

## 1. Apex
ARG APEX_REVISION=SKIP
ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
ARG APEX_MAX_JOBS=4

RUN if [ "${APEX_REVISION}" != SKIP ]; then \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    echo APEX_REVISION=${APEX_REVISION} && \
    git checkout ${APEX_REVISION} && \
    echo APEX_COMMIT_HASH=$(git rev-parse HEAD) && \
    MAX_JOBS=${APEX_MAX_JOBS} NVCC_APPEND_FLAGS="--threads 8" pip install -v --no-build-isolation --no-cache-dir --disable-pip-version-check --config-settings "--build-option=--cpp_ext --cuda_ext --bnp --xentropy --deprecated_fused_adam --deprecated_fused_lamb --fast_multihead_attn --distributed_lamb --fast_layer_norm --transducer --distributed_adam --fmha --fast_bottleneck --nccl_p2p --peer_memory --permutation_search --focal_loss --fused_conv_bias_relu --index_mul_2d --cudnn_gbn --group_norm" . \
    ; fi

## 2. Transformer Engine
ARG TE_REVISION=v2.2.1
ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    git -C 3rdparty/cudnn-frontend checkout 6ed19fd213e33af2d9a1841b1023ccb2f81d45a1 && \
    git -C 3rdparty/googletest checkout f8d7d77c06936315286eb55f8de22cd23c188571 && \
    NVTE_CUDA_ARCHS="90;100" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps . \
    ; fi

## 3. NeMo
ARG NEMO_REVISION=25.04-alpha.rc2
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}

RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /NEMO_COMMIT_HASH.env && \
    pip uninstall -y nemo-toolkit sacrebleu && \
    pip install "cython<3.0.0" && \
    pip install -e ".[llm]" && \
    pip install -e ".[nlp]"

## 3.1 NeMo-Run
ARG NEMORUN_REVISION=v0.4.0rc2.dev0
ENV CUSTOM_NEMORUN_REVISION ${NEMORUN_REVISION}

RUN git clone https://github.com/NVIDIA/NeMo-Run.git && \
    cd NeMo-Run && \
    git checkout ${NEMORUN_REVISION} && \
    echo NEMORUN_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install -e .

# 4. Megatron-core
ARG MCORE_REVISION=25.04-alpha.rc2
ENV CUSTOM_MCORE_REVISION ${MCORE_REVISION}

RUN if [ "${MCORE_REVISION}" != SKIP ]; then \
    pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM Megatron-LM && \
    cd Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install . && \
    cd megatron/core/datasets && \
    make \
    ; fi

ENV PYTHONPATH "${PYTHONPATH}:/workspace/Megatron-LM"


## 5. Benchmark dependencies
RUN pip uninstall transformers -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Benchmark code
WORKDIR /workspace/llm

COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps .
ENV PYTHONPATH "/workspace/llm:/workspace/NeMo:${PYTHONPATH}"

