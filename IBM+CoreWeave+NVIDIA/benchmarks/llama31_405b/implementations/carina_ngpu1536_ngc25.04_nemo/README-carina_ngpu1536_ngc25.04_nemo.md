## Steps to launch training

### carina_ngpu1536_ngc25.04_nemo

Launch configuration and system-specific hyperparameters for the
carina_ngpu1536_ngc25.04_nemo submission are in the
`benchmarks/llama31_405b/implementations/carina_ngpu1536_ngc25.04_nemo/config_GB200_384x4x48xtp4pp8cp2.sh` script.

Steps required to launch training for carina_ngpu1536_ngc25.04_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB200_384x4x48xtp4pp8cp2.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
