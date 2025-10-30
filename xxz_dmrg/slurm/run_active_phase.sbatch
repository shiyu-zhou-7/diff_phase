#!/bin/bash
#SBATCH --job-name=xxz_active
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/xxz_active_%j.out
#SBATCH --error=logs/xxz_active_%j.err

cd ~/diffphase/diff_phase
git fetch origin main
git sparse-checkout init --cone
git sparse-checkout set xxz_dmrg
git checkout main
git pull origin main

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jaxgpu

cd xxz_dmrg
python main_active_phase.py

