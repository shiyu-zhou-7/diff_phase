#!/bin/bash
#SBATCH --job-name=z2gauge_ap_gpu         # descriptive job name
#SBATCH --partition=h200q                    # GPU H200 queue
#SBATCH --gres=gpu:1                         # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    # CPU cores for preprocessing / JAX threads
#SBATCH --mem=64G                            # total system RAM
#SBATCH --time=1-00:00:00                    # 1 day walltime
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_ap_gpu_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_ap_gpu_%j.err

# Per-run knobs picked up from env (defaults in main_active_phase.py if unset):
#   H_INIT           : starting h                            (default -1.0)
#   LX, LY           : lattice sizes                         (defaults 3, 3 ; existing AE checkpoint is Lx2Ly3)
#   MAX_ITERS        : outer iters                           (default 10)
#   HAM_LR           : h-Adam step                           (default 1e-4 from HamConfig)
#   MAX_STEPS_BLOCK  : max inner h-steps per outer iter      (default 2000 from HamConfig)
#   PARAM_TOL_CHANGE : early-stop tolerance on |h_change|    (default 5e-5; set 0 to disable)
#
# Exactly 100 inner h-steps at lr=0.1 (no early stall break, no outer-iter loop):
#   for h in -1.5 -1.0 -0.8; do
#       sbatch --export=ALL,H_INIT=$h,LX=2,LY=3,HAM_LR=0.1,MAX_STEPS_BLOCK=100,MAX_ITERS=1,PARAM_TOL_CHANGE=0 \
#              gpu_run_active_phase.sbatch
#   done

set -xeuo pipefail

# === Load environment ===
module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# === Git pull ===
cd /home/szhou3/diffphase/diff_phase
git fetch origin main
# git sparse-checkout init --cone
git sparse-checkout set tfim_reorg z2gauge xxz_dmrg
git checkout main
git pull origin main

# === run ===
cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
echo "...Starting z2gauge active phase searching..."
python -u main_active_phase.py
