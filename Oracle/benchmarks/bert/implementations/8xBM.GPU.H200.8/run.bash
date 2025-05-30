export EVALDIR="/mnt/localdisk7/mlperf/bert/hdf5/eval_varlength"
export DATADIR_PHASE2="/mnt/localdisk7/mlperf/bert/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
export DATADIR_PHASE2_PACKED="/mnt/localdisk7/mlperf/bert/packed_data"
#export CHECKPOINTDIR_PHASE1="/mnt/localdisk4/mlperf/bert/phase1"
export CHECKPOINTDIR_PHASE1="/home/ubuntu/sd/bert/checkpoint/phase1"
#export LOGDIR=/mnt/localdisk7/mlperf/bertcode/log # set the place where the output logs will be saved
export LOGDIR=/home/ubuntu/sd/bert/log
export CONT="/mnt/localdisk7/mlperf/bertcode/nvcr.io+nvdlfwea+mlperftv50+bert+20250331.pytorch.sqsh"
#source config_DGXH100_1x8x48x1_pack.sh  # select config and source it
source config_DGXH100_8x8x36x1_pack.sh
#export SLURM_JOB_NODELIST=GPU-477
export NODELIST=GPU-[47,54,97,128,143,161,167,190,199,204,217,223,226,230,232,235,239,240,246,249,257,272,278,286,294,320,323,324,325,333,340,348,352,358,381,386,387,392,404,408,413,418,419,438,446,448,457,458,461,474,475,484,497,498,511,513,515,518,532,545,549,552,562,574,577,580,593,594,604,605,608,610,613,614,615,622,630,637,638,644,647,656,673,676,678,679,682,688,689,701,712,713,715,719,722,725,726,753,761,764,767,768,789,790,798,799,809,818,823,826,828,833,834,853,854,870,874,875,886,887,891,892,903,924,925,926,927,931,937,940,941,945,946,973,987,999,1006,1010,1019]
#export NODELIST=GPU-477
echo ${DGXNNODES}
export NEXP=1
sbatch -N ${DGXNNODES} --wait --nodelist $NODELIST  --time=${WALLTIME} run.sub 
