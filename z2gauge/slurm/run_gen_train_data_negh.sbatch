#!/bin/bash
#SBATCH --job-name=z2gauge_gen_negh
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_gen_negh_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_gen_negh_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=longq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# CPU-only JAX (no GPU needed for ITE on the 4096-dim Hilbert space)
export JAX_PLATFORMS=cpu

# Load conda (use the system-wide init; `source ~/.bashrc` doesn't reliably
# expose `conda` in a non-interactive sbatch shell).
module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# === run ===
cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge negh data generation..."
python -u generate_train_data_negh.py
