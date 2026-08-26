#!/bin/bash
#SBATCH -J fin100000
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-49
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH -o /scratch/yuxzhang/final/f100000_%a.log
module -q load gcc python
cd /scratch/yuxzhang/final
DS=(20 40 60 80 100 120 140 160 180 200)
i=$SLURM_ARRAY_TASK_ID
export SW_D=${DS[$((i % 10))]}
export SW_SEED=$((i / 10))
export SW_LAZY=30 SW_BUDGET=100000 SW_START=e0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
echo "task $i : d=$SW_D budget=$SW_BUDGET seed=$SW_SEED lazy=30 e0 L=500"
/scratch/yuxzhang/sptenv/bin/python -u jed_final.py
