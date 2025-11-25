#!/bin/bash
#SBATCH --job-name=z2ham
#SBATCH --output=z2ham.out
#SBATCH --error=z2ham.err
#SBATCH --time=02:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Always force JAX to use CPU
export JAX_PLATFORMS=cpu

# Load conda
source ~/.bashrc
conda activate test

# Run the script
python -u ~/diffphase/diff_phase/z2gauge/scripts/hamiltonian/z2ham.py
