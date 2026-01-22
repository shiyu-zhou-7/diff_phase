#!/bin/bash
#SBATCH --job-name=z2gauge_ap
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_ap_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_ap_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=longq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Always force JAX to use CPU
export JAX_PLATFORM_NAME=cpu

# Load conda
module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# === run ===
cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge active phase searching..."
python -u main_active_phase.py
