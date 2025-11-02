#!/bin/bash
#SBATCH --job-name=xxz_dmrg_active
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.err

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh

cd ~/diffphase/diff_phase
git fetch origin main
git sparse-checkout init --cone
git sparse-checkout set xxz_dmrg
git checkout main
git pull origin main

cd xxz_dmrg/scripts
python main_active_phase.py

