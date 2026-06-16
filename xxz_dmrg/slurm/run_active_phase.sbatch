#!/bin/bash
#SBATCH --job-name=xxz_dmrg_active
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.err
#SBATCH --signal=USR1@120

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

# make sure the log dir exists
mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
# Serialize the git step: concurrent seed jobs share this checkout dir and
# would otherwise race on .git/index.lock. All jobs pull the same commit.
(
  flock 9
  git fetch origin main
  git sparse-checkout set tfim_reorg z2gauge xxz_dmrg
  git checkout main
  git pull origin main
) 9>/home/szhou3/diffphase/.gitsync.lock


cd /home/szhou3/diffphase/diff_phase/xxz_dmrg/scripts/
export JAX_PLATFORMS=cpu
echo "AE_SEED=${AE_SEED:-<default>} INIT_DELTA=${INIT_DELTA:-<default>} INIT_H=${INIT_H:-<default>} FRESH=${FRESH:-<default>}"
python -u main_active_phase.py

