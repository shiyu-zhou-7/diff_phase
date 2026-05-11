#!/bin/bash
#SBATCH --job-name=tfim_active
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/tfim_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/tfim_active_%j.err
#SBATCH --signal=USR1@120

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# make sure the log dir exists
mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
git fetch origin main
git sparse-checkout set tfim_reorg
git checkout main
git pull origin main

cd /home/szhou3/diffphase/diff_phase/tfim_reorg/scripts/
export JAX_PLATFORMS=cpu
# H_INIT overridable at submission: H_INIT=-0.3 sbatch --export=ALL ...
export H_INIT=${H_INIT:--1.5}
echo "[sbatch] running main_active_phase.py with H_INIT=${H_INIT}"
python -u main_active_phase.py
