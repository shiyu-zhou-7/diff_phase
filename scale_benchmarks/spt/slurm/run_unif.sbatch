#!/bin/bash
#SBATCH -J unifBase
#SBATCH -A phys-ge-tanq-erc
#SBATCH -p standard
#SBATCH --array=0-29
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH -o /scratch/yuxzhang/final/u_%a.log
module -q load gcc python
cd /scratch/yuxzhang/final
DS=(20 40 60 80 100 120 140 160 180 200)
BS=(30000 100000 300000)
i=$SLURM_ARRAY_TASK_ID
export SW_D=${DS[$((i % 10))]}
export SW_BUDGET=${BS[$(( (i / 10) % 3 ))]}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "uniform baseline d=$SW_D N=$SW_BUDGET (5 realisations)"
/scratch/yuxzhang/sptenv/bin/python -u unif_base.py
