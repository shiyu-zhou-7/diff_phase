#!/bin/bash
#SBATCH -J chernRELAB
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-49
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/yuxzhang/chern/relab_%a.log
module -q load gcc python
cd /scratch/yuxzhang/chern
export M_CONV=768 M_CHECK=1536 NCHUNK=10 VAL_Q=300000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONFAULTHANDLER=1
/scratch/yuxzhang/sptenv/bin/python -u chern_relabel.py
