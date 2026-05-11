#!/bin/bash
#SBATCH --job-name=z2gauge_ae
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_ae_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_ae_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=longq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Per-run knobs picked up from env (defaults in main_train_ae.py if unset):
#   LX, LY     : lattice sizes               (defaults 3, 3)
#   DATA_PATH  : training-data pkl           (default: ../data/data_ite_confined_{Lx}x{Ly}_n1000.pkl)
#   PHASE_TAG  : output filename suffix      (default 'params' for back-compat)
#
# Train two phase-specific AEs in parallel:
#   sbatch --export=ALL,LX=2,LY=3,DATA_PATH=../data/data_ite_confined_2x3_h-1.0_to_-0.4_n1000.pkl,PHASE_TAG=confined   run_ae.sbatch
#   sbatch --export=ALL,LX=2,LY=3,DATA_PATH=../data/data_ite_deconfined_2x3_h-0.2_to_-0.001_n1000.pkl,PHASE_TAG=deconfined run_ae.sbatch

# Always force JAX to use CPU
export JAX_PLATFORMS=cpu

# Load conda
module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# === run ===
cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge autoencoder training..."
python -u main_train_ae.py
