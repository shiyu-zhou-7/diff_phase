#!/bin/bash
#SBATCH --job-name=z2gauge_data_gen
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_data_gen_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_data_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Always force JAX to use CPU
export JAX_PLATFORMS=cpu

# Load conda
source ~/.bashrc
conda activate test

# === run ===
cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge autoencoder training..."
python -u generate_confined_data.py
