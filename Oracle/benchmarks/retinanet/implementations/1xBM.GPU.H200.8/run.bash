#export DATADIR="/mnt/localdisk1/mlperf/data/open-images-v6"
export DATADIR="/mnt/localdisk5/mlperf/ssd/data/open-images-v6"
#export BACKBONE_DIR="/mnt/localdisk1/mlperf/data/torch-home/hub/checkpoints"
export BACKBONE_DIR="/home/ubuntu/sd/ssd/checkpoints/checkpoints"
#export LOGDIR="/mnt/localdisk1/mlperf/log"
export LOGDIR="/home/ubuntu/sd/ssd/log"
export CONT=/mnt/localdisk1/mlperf/ssd/nvcr.io+nvdlfwea+mlperftv50+ssd+20250331.pytorch.sqsh
#source config_DGXH100_008x08x004.sh
source config_DGXH100_001x08x032.sh
#export SLURM_JOB_NODELISTGPU=GPU-[294,798,799,278,320,818,826,324,286,823,853,833,323,333,352,325,348,809,834,828,854,870,892,340,358,891,887,392,874,413,386,925,404,446,875,419,886,381,387,903,924,927,474,408,418,448,461,937,457,945,438,926,987,931,940,458,973,475,999,941,946,1010,497,484,1019,498,552,518,491,511,515,545,549,513,1006,605,574,532,580,593,562,614,608,630,604,594,610,577,128,644,638,622,613,615,47,637,679,54,656,97,161,676,673,143,678,199,682,688,647,712,689,719,167,190,204,761,722,713,701,217,753,232,768,230,246,226,715,240,790,789,725,257,272,239,223,235,764,726,767,249]
export SLURM_JOB_NODELISTGPU=GPU-491
#export SLURM_JOB_NODELISTGPU=GPU-[47,419,498,753,790,892,903,1006]
export SLURM_MPI_TYPE=pmi2
export NCCL_NVLS_ENABLE=1
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0
set -eux
export PMI_DEBUG=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_btl_tcp_if_include="10.224.0.0/12"
export PMIX_MCA_gds="^ds12" \
      NCCL_SOCKET_NTHREADS=16 \
      NCCL_DEBUG=WARN \
      NCCL_CUMEM_ENABLE=0 \
      NCCL_IB_SPLIT_DATA_ON_QPS=0 \
      NCCL_IB_QPS_PER_CONNECTION=1 \
      NCCL_IB_GID_INDEX=3 \
      NCCL_IB_TC=41 \
      NCCL_IB_SL=0 \
      NCCL_IB_TIMEOUT=22 \
      NCCL_NET_PLUGIN=none \
      NCCL_SOCKET_IFNAME=eth0 \
      NCCL_IGNORE_CPU_AFFINITY=1 \
      RX_QUEUE_LEN=8192 \
      IB_RX_QUEUE_LEN=8192 \
      UCX_NET_DEVICES=eth0 \
      UCX_TLS=tcp \
      HCOLL_ENABLE_MCAST_ALL=0 \
      coll_hcoll_enable=0 \
      NCCL_IB_HCA='=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11'

export NEXP=1
sbatch --wait -N $DGXNNODES --nodelist $SLURM_JOB_NODELISTGPU -t $WALLTIME run.sub  # you may be required to set --account and --partition here
