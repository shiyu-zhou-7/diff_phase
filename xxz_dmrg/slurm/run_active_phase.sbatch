#!/bin/bash
#SBATCH --job-name=xxz_dmrg_active
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/xxz_dmrg_active_%j.err

set -xeuo pipefail

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate jaxgpu

# make sure the log dir exists
mkdir -p /home/szhou3/diffphase/logs


echo "== whoami =="
whoami
echo "== hostname =="
hostname
echo "== pwd BEFORE cd =="
pwd

cd /home/szhou3/diffphase/diff_phase

echo "== pwd AFTER cd =="
pwd
echo "== ls here =="
ls -R | head -200

git fetch origin main
git sparse-checkout init --cone
git sparse-checkout set xxz_dmrg
git checkout main
git pull origin main

echo "== which python =="
which python
python -c "import sys; print('python version:', sys.version)"

cd /home/szhou3/diffphase/diff_phase/scripts
ls -l
python main_active_phase.py

