#!/bin/bash
#SBATCH --job-name=spt_benchmark
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/spt_benchmark_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/spt_benchmark_%j.err
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
# everything under it -- including spt/data and spt/figures. benchmark.py writes to
# the relative paths ../data and ../figures, so we run it from a persistent results
# dir whose parent holds those folders, well outside the tree git can wipe.
RESULTS_ROOT=/home/szhou3/diffphase/spt_results
mkdir -p "$RESULTS_ROOT/data" "$RESULTS_ROOT/figures" "$RESULTS_ROOT/cwd"
cd "$RESULTS_ROOT/cwd"   # ../data, ../figures -> $RESULTS_ROOT/{data,figures}

# CPU-only; cap threads so JAX does not oversubscribe on the small (2^{L-1})
# sector matrices (oversubscription is a big slowdown there).
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# Headline d=8 benchmark. All knobs are env-overridable at submit time:
#   sbatch --export=ALL,D=8,K=24,L=10,EPOCHS=2000 spt/slurm/run_benchmark.sbatch
export D=${D:-8}
export K=${K:-24}
export L=${L:-10}
export EPOCHS=${EPOCHS:-2000}
export MINI_EPOCHS=${MINI_EPOCHS:-1000}
export INIT_N=${INIT_N:-300}
export BOOTSTRAP_N=${BOOTSTRAP_N:-150}
export MAX_OUTER=${MAX_OUTER:-4}
export EPS=${EPS:-0.1}
export SEED=${SEED:-0}

echo "spt benchmark  D=$D K=$K L=$L EPOCHS=$EPOCHS MAX_OUTER=$MAX_OUTER SEED=$SEED"
echo "writing results to $RESULTS_ROOT/{data,figures}"
python -u /home/szhou3/diffphase/diff_phase/spt/scripts/benchmark.py
