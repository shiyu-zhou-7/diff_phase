#!/bin/bash
#SBATCH -J fin30k
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-49
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 03:00:00
#SBATCH -o /scratch/yuxzhang/final/f30k_%a.log
module -q load gcc python
cd /scratch/yuxzhang/final
export TMPDIR=/scratch/yuxzhang/final/tmp30_$SLURM_ARRAY_TASK_ID
mkdir -p $TMPDIR
DS=(20 40 60 80 100 120 140 160 180 200)
i=$SLURM_ARRAY_TASK_ID
export SW_D=${DS[$((i % 10))]}
export SW_SEED=$((i / 10))
export SW_LAZY=30 SW_BUDGET=30000 SW_START=e0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
echo "task $i : d=$SW_D budget=30000 seed=$SW_SEED lazy=30 e0 L=500"
/scratch/yuxzhang/sptenv/bin/python -u jed_final.py
