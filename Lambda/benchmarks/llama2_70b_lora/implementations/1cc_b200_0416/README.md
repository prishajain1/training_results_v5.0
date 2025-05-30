# MLPerf Training v5.0

## Usage

See [See Extra config](#extra-config-for-1cc-b200) before moving forward.

### Dataset preparation

Follow guide from NVIDIA's [mlperftv50](https://registry.ngc.nvidia.com/orgs/nvdlfwea/teams/mlperftv50/containers/llama2_70b_lora-amd) drop.


```
# From worker node
docker login nvcr.io

export CONT=nvcr.io/nvdlfwea/mlperftv50/llama2_70b_lora-amd:20250416
docker pull $CONT

# change this to your own path (shared fs, local storage etc)
export STORAGETPATH=/data
export DATAPATH=$STORAGETPATH/mlperf/data/llama2_70b_lora
mkdir -p $DATAPATH

docker run -it --rm --gpus all --network=host --ipc=host --volume $DATAPATH:/data $CONT

# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download and preprocess dataset; takes less than 1 minute
python scripts/download_model.py --model_dir /data/model  # download and preprocess model checkpoint used for initialization; could take up to 30 minutes
```

### Configure file

Make your own config files for different sizes of runs. e.g.

```
# 1x node
# Notice ENROOT_CONFIG_PATH is set to ~/.config/enroot
config_1ccB200_1x8x1xtp1pp1cp1.sh
```

### Run benchmarks

From the headnode
```
export STORAGETPATH=/data # your shared fs or local storage
cd $STORAGETPATH/mlcommon_training_5.0/Lambda/benchmarks/llama2_70b_lora/implementations/1cc_b200_0416
export DATADIR=$STORAGETPATH/mlperf/data/llama2_70b_lora/gov_report 
export MODEL=$STORAGETPATH/mlperf/data/llama2_70b_lora/model  
export SLURMLOGDIR=slurm_outputs/$(hostname)
export CONT=nvcr.io/nvdlfwea/mlperftv50/llama2_70b_lora-amd:20250416 

mkdir -p $SLURMLOGDIR
```

1xnode run
```
source config_1ccB200_1x8x1xtp1pp1cp1.sh 
export DGXNNODES=1
export LOGDIR="$STORAGETPATH/mlcommon_training_5.0/Lambda/benchmarks/llama2_70b_lora/implementations/1cc_b200_0416/logs/$(hostname)/n$DGXNNODES/$(date '+%Y%m%d_%H%M%S')"
mkdir -p $LOGDIR
sbatch -N $DGXNNODES -t $WALLTIME -o $SLURMLOGDIR/%x_%j.out run.sub
```

The results will be saved in `$LOGDIR`. The `compliance_timestamp.out` should have _pass_ for successful runs. 


## Extra config for 1cc B200

### Enroot config

Ensure following credentials are in place to ensure the pyxis+enroot can pull the containers from nvcr.io docker registry during a training job. 

`mkdir -p ~/.config/enroot`<br>
`# add the following line to ~/.config/enroot/.credentials`<br>
`machine nvcr.io login $oauthtoken password YOUR_NGC_API_KEY`<br>
`# Ensure propoer permissions`<br>
`chmod 600 ~/.config/enroot/.credentials`<br>

You might want to add `export ENROOT_CONFIG_PATH=<the-above-path>` to benchmark config files so enroot will look at the correct place for credentials. 

### Pyxis + nvidia fabricmanager socket: Permission denied

Pyxis trying to start the container require access to nvidia fabricmanager sockets under /var/run/nvidia-fabricmanager/socket, based on how the system was set access to this path could be restricted. Below here is a quick workaround to this issue, although not for a production deployment. 

`sudo chmod 755 /var/run/nvidia-fabricmanager`<br>
`sudo chmod a+rw /var/run/nvidia-fabricmanager/fm_sm_ipc.socket`<br>
`sudo systemctl restart nvidia-fabricmanager`<br>
