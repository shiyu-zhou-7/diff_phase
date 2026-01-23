#!/bin/bash
#SBATCH --job-name=z2gauge_ap
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_ap_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_ap_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=longq
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# Always force JAX to use CPU
export JAX_PLATFORM_NAME=cpu

echo "==== ulimit ===="
ulimit -a
echo "==== meminfo ===="
grep -E "MemTotal|MemFree|MemAvailable" /proc/meminfo

export JAX_PLATFORM_NAME=cpu
export JAX_PLUGINS=""
export JAX_LOG_COMPILES=1

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


# Load conda
module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# === run ===
cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge active phase searching..."
python -u main_active_phase.py
