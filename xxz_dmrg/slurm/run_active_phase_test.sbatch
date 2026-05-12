#!/bin/bash
#SBATCH --job-name=xxz_dmrg_active_test
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/xxz_dmrg_active_test_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/xxz_dmrg_active_test_%j.err

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# make sure the log dir exists
mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
git fetch origin main
git sparse-checkout set tfim_reorg z2gauge xxz_dmrg
git checkout main
git pull origin main


cd /home/szhou3/diffphase/diff_phase/xxz_dmrg/scripts/
export JAX_PLATFORMS=cpu
python -u main_active_phase_test.py
