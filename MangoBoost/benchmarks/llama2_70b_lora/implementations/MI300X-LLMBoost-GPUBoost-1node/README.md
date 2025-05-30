# MangoBoost MLPerf Training v5.0 Benchmark
This folder contains all of the code necessary to run:

+ MLPerf Training MI300X Single-Node (8 GPUs on a single 8xMI300X AMD)
+ MLPerf Training MI300X Two-Node (16 GPUs across two 8xMI300X AMD and Mango GPUBoost RDMA DPUs)
+ MLPerf Training MI300X Four-Node (32 GPUs across four 8xMI300X AMD and Mango GPUBoost RDMA DPUs)

We benchmark on LLama2-70B to generate all the results. For the software, we use **LLMBoost-1.1.0** to enhance the performance, scalability and ease deployment. Please also ensure that all other software dependecies: ROCm 6.4.0, Python 3.10 and RCCL 2.22.3 are installed. 

For the hardware, we runs the benchmark on 1/2/4 nodes AMD Instinct MI300X 192GB HBM3 GPUs. On each GPU node, we have installed 8x **MangoBoost GPUBoost RDMA** to handle multi-node communication.

For single-node, a performance of 29.60 minutes is observed using a single node of 8x MI300X GPUs. 16.36 minutes and 10.92 minutes of performance are observed on two-node and four-node setups.

The following steps outline the process of setting up a Docker environment and the details to reproduce our MLPerf V5.0 Training result. 

# 1. Setup Docker Image

Run the following build command from the root of the repository. The build process will take a while to complete.

```bash
docker build -t rocm/amd-mlperf:llama2_70b_training_5.0 .
```
# 2. Prepare Dataset

## General Information

GovReport is a dataset for long document summarization that consists of reports written by government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has length 8192.

The used model is the LLama2-70B with fused QKV. You will need 270GB to download and convert the model.

## Download and Preprocess Data

Start the docker container by mounting the volume you want to use for downloading the data under `/data` within the container. In this example we use `/models/amd2025_april/mlperf_llama2` as the host download directory:

```bash
docker run -it -v /models/amd2025_april/mlperf_llama2:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_5.0
```

Start the script for downloading and preprocessing data from within the container:

```bash
bash ./scripts/prepare_data_and_model.sh
```
## Verify Data

The data and model files are stored under `/data` within the container.  
After preprocessing, you should see the following files in the `/data/model` directory:
```
<hash>_tokenizer.model  llama2-70b.nemo  model_config.yaml  model_weights
```
And the following files in the `/data/data` directory:
```
train.npy  validation.npy
```
## Exit Container 

Exit the container by running the below command

```bash
exit
```

# 3. Running the Benchmark 

### 3.1. Prepare the Docker Environment on Each Node 
    
    # Pull the docker image
    docker pull llmboost/mb-llmboost-training:mlperf-5.0-prod

    # Start the Docker container
    docker run --rm -it \
        --network host \
        --ipc host \
        --uts host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --group-add video \
        --device /dev/dri:/dev/dri \
        --device /dev/kfd:/dev/kfd \
        -v /models/amd2025_april/mlperf_llama2/data:/data \
        -v /models/amd2025_april/mlperf_llama2/model:/ckpt \
        -w /workspace/mlperf_training \
        llmboost/mb-llmboost-training:mlperf-5.0-prod

### 3.2. Running the MLPerf Benchmark

> Note: In multi-node runs, you must replace `MASTER_ADDR` with the IP address of the main node (typically `node-0`) in your cluster. Also, please modify the `export NCCL_SOCKET_IFNAME=ens11np0,ens12np0,ens21np0,ens22np0,ens31np0,ens32np0,ens41np0,ens42np0` inside `config_MI300X_*x8x1.sh` according to your network cards setup.

#### Single Node Benchmark
    llmboost mlperf --config_sh config_MI300X_1x8x1.sh 2>&1 | tee "log_single_node.txt"

#### Multi-Node Benchmark (2-nodes)

    # On node-0
    llmboost mlperf --MASTER_ADDR 10.4.16.1 --RANK 0 --config_sh config_MI300X_2x8x1.sh 2>&1 | tee "log_2_nodes.txt"

    # On node-1
    llmboost mlperf --MASTER_ADDR 10.4.16.1 --RANK 1 --config_sh config_MI300X_2x8x1.sh


#### Multi-Node Benchmark (4-nodes)

    # On node-0
    llmboost mlperf --MASTER_ADDR 10.4.16.1 --RANK 0 --config_sh config_MI300X_4x8x1.sh 2>&1 | tee "log_4_nodes.txt"

    # On node-1
    llmboost mlperf --MASTER_ADDR 10.4.16.1 --RANK 1 --config_sh config_MI300X_4x8x1.sh
    
    # On node-2
    llmboost mlperf --MASTER_ADDR 10.4.16.1 --RANK 2 --config_sh config_MI300X_4x8x1.sh
    
    # On node-3
    llmboost mlperf --MASTER_ADDR 10.4.16.1 --RANK 3 --config_sh config_MI300X_4x8x1.sh


# 4. Quality
### Quality metric
Cross entropy loss
### Quality target
0.925
### Evaluation frequency
Every 384 samples
### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples
