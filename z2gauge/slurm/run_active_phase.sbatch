#!/bin/bash
#SBATCH --job-name=z2gauge_active
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/z2gauge_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/z2gauge_active_%j.err
#SBATCH --signal=USR1@120

# Per-run knobs picked up from env (defaults in main_active_phase.py if unset):
#   H_INIT       starting h                          (default -0.4)
#   SEED         RNG seed                            (default HamConfig.seed)
#   LX, LY       lattice sizes                       (default 2, 3)
#   EPOCHS       AE training epochs                  (default AEConfig.epochs)
#   MAX_OUTER    outer-loop cap                      (default 5)
#   INIT_N       initial-circle sample count         (default 200)
#   BOOTSTRAP_N  per-circle sample count on retrain  (default 100)
#
# Examples:
#   sbatch --export=ALL,H_INIT=-0.5 run_active_phase.sbatch
#   sbatch --export=ALL,H_INIT=-0.1,EPOCHS=200,INIT_N=20,BOOTSTRAP_N=15,MAX_OUTER=1 run_active_phase.sbatch  # smoke

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
git fetch origin main
# Additive sparse-checkout: keep all three model subtrees so parallel jobs
# from other models don't get their working tree wiped (see commit cf1b93f).
git sparse-checkout set tfim_reorg z2gauge xxz_dmrg
git checkout main
git pull origin main

cd /home/szhou3/diffphase/diff_phase/z2gauge/scripts/
export JAX_PLATFORMS=cpu
export H_INIT=${H_INIT:--0.4}
echo "[sbatch] running main_active_phase.py with H_INIT=${H_INIT}"
python -u main_active_phase.py
