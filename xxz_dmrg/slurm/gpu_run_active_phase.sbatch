#!/bin/bash
#SBATCH --job-name=active_phase_gpu          # descriptive job name
#SBATCH --partition=h200q                    # GPU H200 queue
#SBATCH --gres=gpu:1                         # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    # CPU cores for preprocessing / JAX threads
#SBATCH --mem=64G                            # total system RAM
#SBATCH --time=1-00:00:00                    # 1 day walltime
#SBATCH --output=/home/szhou3/diffphase/logs/active_phase_gpu_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/active_phase_gpu_%j.err

set -xeuo pipefail

# === Load environment ===
module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# === Git pull ===
cd /home/szhou3/diffphase/diff_phase
git fetch origin main
git sparse-checkout init --cone
git sparse-checkout set xxz_dmrg
git checkout main
git pull origin main

# === run ===
cd /home/szhou3/diffphase/diff_phase/xxz_dmrg/scripts
python main_train_ae.py
