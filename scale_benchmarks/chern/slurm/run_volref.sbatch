#!/bin/bash
#SBATCH -J chernVOL
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-19
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/yuxzhang/chern/volref_%a.log
module -q load gcc python
cd /scratch/yuxzhang/chern
export M_CONV=768 VOL_N=1000000 VOL_TASKS=20
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONFAULTHANDLER=1
/scratch/yuxzhang/sptenv/bin/python -u chern_volref.py
