# Instruction for StableDiffusion MLPerf workload

## 1. Problem

Stable Diffusion v2 Model

### Requirements

*   [Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
*   [GKE (Google Kubernetes Engine)](https://cloud.google.com/kubernetes-engine)

## 2. Directions

### Dependency Setup
Build a maxdiffusion docker image or use a prebuilt docker image [here](https://pantheon.corp.google.com/artifacts/docker/cloud-tpu-multipod-dev/us/gcr.io/maxdiffusion_mlperf_5_0?e=13802955&mods=logs_tg_test) (request to access).

If you'd like to build a maxdiffusion docker image:

```
# clone maxdiffusion
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion
git checkout mlperf_5.0
bash docker_build_dependency_image.sh JAX_VERSION=0.6.0
```
This will build a local image named `maxdiffusion_base_image`.

### Cluster Setup
We use [xpk](https://github.com/AI-Hypercomputer/xpk) to create the cluster.

```bash
bash setup.sh
```

### Steps to launch training
We use [xpk](https://github.com/AI-Hypercomputer/xpk) to deploy jobs as well.
If you created your own docker image in `Dependency Setup`, update the
script so that the `--base-docker-image` points to that image:

```
--base-docker-image=maxdiffusion_base_image
```

Then launch the training:

```
bash trillium-256.sh
```

## 3. Dataset

Please refer to the
[instructions](https://github.com/mlcommons/training/tree/master/stable_diffusion)
from the reference to download the dataset.

We preprocessed the data:
##### 1. Download the dataset as instructed [here](https://github.com/mlcommons/training/tree/master/stable_diffusion#the-datasets)
##### 2. Create a persistent disk and attach to VM as read-write.
##### 3. Create 2 directories inside the persistent disk to store the extracted files and created tfrecords.
##### 4. Run this [file](https://github.com/AI-Hypercomputer/maxdiffusion/blob/mlperf_4/src/maxdiffusion/pedagogical_examples/to_tfrecords.py) to preprocess and pre-encode text embedding:
```
python to_tfrecords.py \
  src/maxdiffusion/configs/base_2_base.yml attention=dot_product \
  data_files_pattern=/mnt/data/webdataset-moments-filtered/*.tar \
  extracted_files_dir=/tmp/raw-data-extracted \
  tfrecords_dir=/mnt/data/tf_records_512_encoder_state_fp32 \
  run_name=test no_records_per_shard=12720 base_output_directory=/tmp/output
```
##### 5. uploaded to your gcs bucket location `gs://your_bucket/laion400m/raw_data/tf_records_512_encoder_state_fp32`


## 4. Model

The model largely follows the Stable Diffusion v2 reference paper, with key
model architecture configs refer
[here](https://github.com/mlcommons/training/tree/master/stable_diffusion#the-model)

### List of Layers

The model largely follows the Stable Diffusion v2 reference paper. refer to
[here](https://github.com/mlcommons/training/tree/master/stable_diffusion#the-model)

## 5. Quality

### Evaluation frequency

Every 512,000 images, or CEIL(512000 / global_batch_size) if 512,000 is not
divisible by GBS.

### Evaluation thoroughness

Evaluation on the validation subset that consists of 30,000 examples subset of
coco-2014. ref to
[here](https://github.com/mlcommons/training/tree/master/stable_diffusion#evaluation-thoroughness)


## 6. Additional notes

####Postprocess for MLLOG from raw run

```
cat ${job_dir}/sd_worker.log | grep MLLOG  > ${job_dir}/result_0.txt
```

####Padded coco 30k val dataset for distributed evaluation

val2014_30k.tsv is padded with last element from 30,000 to 30,720 for
evenly-distributed loading. However filled elements are discarded during eval.