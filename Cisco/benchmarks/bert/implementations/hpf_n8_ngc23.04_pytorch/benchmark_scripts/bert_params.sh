# Debug variables - Hyperfabric
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export UCX_NET_DEVICES="gpu0_eth,gpu1_eth,gpu2_eth,gpu3_eth,gpu4_eth,gpu5_eth,gpu6_eth,gpu7_eth"
export NCCL_IB_HCA="gpu0_rdma,gpu1_rdma,gpu2_rdma,gpu3_rdma,gpu4_rdma,gpu5_rdma,gpu6_rdma,gpu7_rdma"
export NCCL_SOCKET_IFNAME="gpu0_eth,gpu1_eth,gpu2_eth,gpu3_eth,gpu4_eth,gpu5_eth,gpu6_eth,gpu7_eth"
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=8

export BASEDIR="/mnt/nfsshare/images/bert"
export CONT="${BASEDIR}/bert_latest+20241129_dev2.sqsh"

export DATAPATH="/home/hpf/mlperf_data/bert"
export DATAPATH="/mnt/vast/mlperf_data/bert/"
export DATADIR="${DATAPATH}/packed_data"
export EVALDIR="${DATAPATH}/hdf5/eval_varlength/"
export DATADIR_PHASE2="${DATADIR}"
export CHECKPOINTDIR_PHASE1="${DATAPATH}/phase1/"
export CHECKPOINTDIR="${CHECKPOINTDIR_PHASE1}"

export MLPERF_SYSTEM_NAME="HPF"
export MLPERF_CLUSTER_NAME="Cisco AI Cluster"
export SLURM_MPI_TYPE="pmix"
export NCCL_TEST=0

# Number of BERT training iterations to run
export NEXP=10

# Network topology file
export NETWORK_TOPOLOGY_JSON=switches_nexus.json

# Log directory for benchmark logs
export OUTPUT_DIR=/mnt/nfsshare/slurm_outputs
