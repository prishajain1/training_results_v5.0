### Hyperfabric Routing BERT Benchmark Scripts

This folder contains scripts to run BERT on a variety of logical partitions in the Cisco AI Cluster

All of them follow the naming convention:

run_bert_{*num_of_nodes*}node_{*partition*}_{*traffic_pattern*}.sh

Example:

If we want to run BERT with a intra-node traffic pattern on Zone1 with 8 nodes in each zone we can run the following script:
```
./run_bert_8node_zone1_intra_node.sh
```

## Benchmark Parameters
Some parameters like the central location for benchmark logs, network topology json, number of experiments can be managed in
```
bert_params.sh
```
Typical changes would be the following variables:

# Number of BERT training iterations to run
NEXP=10

# Network topology file
NETWORK_TOPOLOGY_JSON=switches_nexus.json

# Log directory for benchmark logs
OUTPUT_DIR=/mnt/nfsshare/slurm_outputs