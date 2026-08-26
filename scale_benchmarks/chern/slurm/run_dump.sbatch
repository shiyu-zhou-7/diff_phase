#!/bin/bash
#SBATCH -J chernDUMP
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-4
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/yuxzhang/chern/dump_%a.log
module -q load gcc python
cd /scratch/yuxzhang/chern
export CH_H=16 VAL_Q=300000
export VAL_SEED=$((200 + SLURM_ARRAY_TASK_ID))
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
echo "dump: w2 dim=50 budget=300k seed=$VAL_SEED (recording paid theta)"
/scratch/yuxzhang/sptenv/bin/python -u chern_dump.py
