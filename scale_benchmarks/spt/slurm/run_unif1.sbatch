#!/bin/bash
#SBATCH -J unif1
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-34
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -o /scratch/yuxzhang/final/u1_%a.log
module -q load gcc python
cd /scratch/yuxzhang/final
CELLS=("180 100000" "200 100000" "120 300000" "140 300000" "160 300000" "180 300000" "200 300000")
i=$SLURM_ARRAY_TASK_ID
c=${CELLS[$((i % 7))]}
export SW_D=$(echo $c | cut -d' ' -f1)
export SW_BUDGET=$(echo $c | cut -d' ' -f2)
export SW_REP=$((i / 7))
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "d=$SW_D N=$SW_BUDGET rep=$SW_REP"
/scratch/yuxzhang/sptenv/bin/python -u unif_one.py
