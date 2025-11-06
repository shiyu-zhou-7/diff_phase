#!/bin/bash
#SBATCH --job-name=xxz_dmrg_active
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.err
#SBATCH --signal=USR1@120
trap "echo 'Signal caught, saving checkpoint'; python save_ckpt.py; exit 0" USR1

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# make sure the log dir exists
mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
git fetch origin main
git sparse-checkout init --cone
git sparse-checkout set xxz_dmrg
git checkout main
git pull origin main


cd /home/szhou3/diffphase/gpu_trial
python test.py

