#!/bin/bash
#SBATCH -J which
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-23
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 03:00:00
#SBATCH -o /scratch/yuxzhang/final/w_%a.log
module -q load gcc python
cd /scratch/yuxzhang/final
ST=(e0 e15 e30 e45 e59 rand)
i=$SLURM_ARRAY_TASK_ID
export SW_D=60 SW_BUDGET=100000 SW_LAZY=30
export SW_START=${ST[$((i % 6))]}
export SW_SEED=$((i / 6))
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
echo "task $i : d=60 start=$SW_START seed=$SW_SEED lazy=30"
/scratch/yuxzhang/sptenv/bin/python -u jed_which.py
