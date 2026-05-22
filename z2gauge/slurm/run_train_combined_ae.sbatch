#!/bin/bash
#SBATCH --job-name=z2gauge_combined_ae
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_combined_ae_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_combined_ae_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=longq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Train a combined AE on the post-fix (new sign convention) confined +
# deconfined ITE datasets. See main_train_combined_ae.py for what it loads
# and why.
#
# Env-var overrides (defaults in main_train_combined_ae.py):
#   LX, LY     : lattice sizes (default 2, 3)
#   SEED       : RNG seed (default AEConfig.seed)
#   CONF_DATA  : confined data pkl path
#   DEC_DATA   : deconfined data pkl path
#
# Typical use:
#   sbatch run_train_combined_ae.sbatch

export JAX_PLATFORMS=cpu

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
git fetch origin main
git sparse-checkout set tfim_reorg z2gauge xxz_dmrg
git checkout main
git pull origin main

cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge combined-AE (postfix) training..."
python -u main_train_combined_ae.py
