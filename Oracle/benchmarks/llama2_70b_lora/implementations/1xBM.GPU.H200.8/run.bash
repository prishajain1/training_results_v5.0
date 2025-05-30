export DATADIR=/mnt/localdisk6/mlperf/llama2_lora/data/gov_report # set correct </path/to/dataset>
export MODEL=/mnt/localdisk6/mlperf/llama2_lora/data/model # set correct </path/to/dataset>
export LOGDIR=/mnt/localdisk6/mlperf/llama2_lora/log/128node # set the place where the output logs will be saved
export CONT=/mnt/localdisk6/mlperf/llama2_lora/cont/sesh+nvdlfwea+mlperftv50+lora+20250331.pytorch.sqsh
export CONT=/mnt/localdisk6/mlperf/llama2_lora/cont/nvcr.io+nvdlfwea+mlperftv50+lora+20250331.pytorch.sqsh
source configs/config_DGXH100_1x8x4xtp4pp1cp1.sh  # select config and source it
#source configs/config_DGXH200_1x8x2xtp1pp1cp2.sh
#source configs/config_DGXH200_64x8x1xtp4pp1cp2.sh
#source configs/config_DGXH200_128x8x1xtp4pp1cp2.sh
#remove 722,682,143
#export NODELIST=GPU-[47,925,927,926,945,386,446,392,404,999,941,419,937,418,931,387,438,408,940,1010,448,973,458,461,457,545,946,549,987,474,484,497,532,518,511,475,513,498,610,1019,552,577,593,515,614,622,608,644,580,605,678,679,656,574,562,630,594,604,637,615,647,712,673,676,128,688,97,689,54,701,753,199,161,715,768,761,167,719,789,713,246,230,190,217,204,764,725,232,240,798,239,294,226,826,799,767,818,272,235,828,833,853,257,249,278,809,324,286,854,348,323,870,333,823,887,320,325,340,891,875,886,358,892,352,874,924,381,903,722]
export NODELIST=GPU-[47,54,97,128,161,167,190,199,204,217,226,230,232,235,239,240,246,249,257,272,278,286,294,320,323,324,325,333,340,348,352,358,381,386,387,392,404,408,413,418,419,438,446,448,457,458,461,474,475,484,497,498,511,513,515,518,532,545,549,552,562,574,577,580,593,594,604,605,608,610,614,615,622,630,637,638,644,647,656,673,676,678,679,688,689,701,712,713,715,719,725,753,761,764,767,768,789,798,799,809,818,823,826,828,833,853,854,870,874,875,886,887,891,892,903,924,925,926,927,931,937,940,941,945,946,973,987,999,1010,1019]
#export NODELIST=GPU-15
export NEXP=10
export WALLTIME=4000
sbatch -N $DGXNNODES --nodelist $NODELIST -t $WALLTIME run.sub  # you may be required to set --account and --partition here
