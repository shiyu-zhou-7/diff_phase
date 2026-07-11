#!/bin/bash
#SBATCH --job-name=spt_active
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/spt_active_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/spt_active_%j.err
#SBATCH --signal=USR1@120

module purge
source /cm/shared/apps/anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate test

mkdir -p /home/szhou3/diffphase/logs

cd /home/szhou3/diffphase/diff_phase
# Serialize the git step (shared checkout dir); add spt to the sparse-checkout set.
(
  flock 9
  git fetch origin main
  git sparse-checkout set tfim_reorg z2gauge xxz_dmrg spt
  git checkout main
  git pull origin main
) 9>/home/szhou3/diffphase/.gitsync.lock

# Outputs must live OUTSIDE the shared git checkout. Sibling xxz/z2 jobs reset the
# sparse-checkout set (excluding spt) at their own startup, which deletes spt/ and
# everything under it -- including spt/data and spt/figures. main_active_phase.py
# writes to the relative paths ../data and ../figures, so we run it from a persistent
# results dir whose parent holds those folders, well outside the tree git can wipe.
RESULTS_ROOT=/home/szhou3/diffphase/spt_results
mkdir -p "$RESULTS_ROOT/data" "$RESULTS_ROOT/figures" "$RESULTS_ROOT/cwd"
cd "$RESULTS_ROOT/cwd"   # ../data, ../figures -> $RESULTS_ROOT/{data,figures}

# CPU-only; cap threads so JAX does not oversubscribe on the small (2^{L-1})
# sector matrices (oversubscription is a big slowdown there).
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# Wavefunction-input active phase discovery. AE = [2^L, 512, 128, 512, 2^L],
# fidelity loss, center_coeff=1e-3, sign-fixed sector-lifted wavefunction input.
# All knobs env-overridable at submit time, e.g.:
#   sbatch --export=ALL,D=5,L=10,MAX_OUTER=6,INIT_N=400 spt/slurm/run_active_phase.sbatch
export D=${D:-5}
export L=${L:-10}
export BC=${BC:-pbc}
export KAPPA=${KAPPA:-0.0}
export SEED=${SEED:-83948}
export EPOCHS=${EPOCHS:-4000}
export MINI_EPOCHS=${MINI_EPOCHS:-2000}
export INIT_N=${INIT_N:-400}
export BOOTSTRAP_N=${BOOTSTRAP_N:-200}
export MAX_OUTER=${MAX_OUTER:-6}

echo "spt active-phase  D=$D L=$L BC=$BC EPOCHS=$EPOCHS MAX_OUTER=$MAX_OUTER SEED=$SEED"
echo "writing results to $RESULTS_ROOT/{data,figures}"
python -u /home/szhou3/diffphase/diff_phase/spt/scripts/main_active_phase.py
