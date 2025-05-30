#!/bin/bash

FS=lvol0
case $FS in
        beeond | lvol | nfsond )
                mkdir -p /$FS/mlperf/
                pushd .
                cd /$FS/mlperf/
		cp -r /hpelustre/SHARED/datasets/MLPERF/training4.1/openimages .
                #pigz -dc /hpelustre/SHARED/datasets/MLPERF/training4.1/openimages.tar.gz | tar xf -
		popd
                export DATADIR="/$FS/mlperf/openimages"
                ;;
        daos )
                srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "export LD_PRELOAD=/usr/lib64/libioil.so"
                mkdir -p /$FS/mlperf/
                pushd .
                cd /$FS/mlperf/
                #tar xf /pfss/hddfs1/MLCOMMONS/training2.1/opeimages.tar
		cp -r /hpelustre/SHARED/datasets/MLPERF/training4.1/openimages .
                popd
                export DATADIR="/$FS/mlperf/openimages"
                ;;
        pfss)
                SBATCH_FS=''
                export DATADIR="/pfss/SHARED/datasets/MLPERF/training4.1/openimages"
               ;;
esac


