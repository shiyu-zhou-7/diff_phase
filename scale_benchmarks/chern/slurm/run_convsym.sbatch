#!/bin/bash
#SBATCH -J chernCONV
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=200,202,204
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/yuxzhang/chern/convsym_%a.log
module -q load gcc python
cd /scratch/yuxzhang/chern
export CONV_N=1000
export VAL_SEED=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONFAULTHANDLER=1
/scratch/yuxzhang/sptenv/bin/python -u chern_convsym.py
