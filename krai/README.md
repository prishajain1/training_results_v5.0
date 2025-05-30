# MLPerf v5.0 Krai Submission

This is a repository of Krai's submission to the MLPerf v5.0 benchmark. It
includes optimized implementations of the benchmark code. The reference
implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v5.0 release

This readme was updated in May. 2025, for the v5.0 round of MLPerf.

# Contents

The implementation(s) in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* Documentation on the dataset, model, and machine setup.

# Running the Benchmark

These benchmarks have been tested on the following machine configuration:

* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using PyTorch 22.09.

Please see [here](./benchmarks/bert/implementations/pytorch-22.09/README.md) for the detail instructions in running the benchmark. 

