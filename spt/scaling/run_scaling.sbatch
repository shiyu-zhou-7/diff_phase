#!/bin/bash
#SBATCH --job-name=spt_scaling
#SBATCH --partition=preq,amdpreq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/home/szhou3/diffphase/logs/spt_scaling_d%x_%j.out
#SBATCH --error=/home/szhou3/diffphase/logs/spt_scaling_d%x_%j.err
#SBATCH --signal=USR1@120
#
# Scaling experiment: query cost of active phase discovery at L=12 (ED) vs d.
# One node-exclusive job per d; K reps with randomized starts run in PARALLEL
# inside the node (preq/amdpreq allocate whole nodes anyway). Submit as:
#   sbatch --export=ALL,D=4  spt/scaling/run_scaling.sbatch
#   sbatch --export=ALL,D=6  spt/scaling/run_scaling.sbatch   # ... etc.
# Each rep: L=12, RANDOM_INIT=1 (uniform on S^{d-1} from its seed), STOP_FULL=1
# (end as soon as all d windings are visited; else run out the hyperparameters).

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

# Outputs live OUTSIDE the shared git checkout (sibling jobs can reset the
# sparse-checkout set and wipe spt/). main_active_phase.py writes to ../data and
# ../figures relative to cwd, so run from a persistent results dir. EXP names
# the experiment subdir (e.g. EXP=longer_d10_armA keeps arms separated).
RESULTS_ROOT=/home/szhou3/diffphase/spt_results/${EXP:-scaling_L12}
mkdir -p "$RESULTS_ROOT/data" "$RESULTS_ROOT/figures" "$RESULTS_ROOT/cwd" "$RESULTS_ROOT/logs"
cd "$RESULTS_ROOT/cwd"   # ../data, ../figures -> $RESULTS_ROOT/{data,figures}

# CPU-only. K reps run concurrently; cap each rep's threads so the node is not
# oversubscribed (K * OMP_NUM_THREADS <= cpus-per-task).
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

D=${D:?set D via sbatch --export=ALL,D=<d>}
export L=${L:-12}
export BC=${BC:-pbc}
K=${K:-8}
BASE_SEED=${BASE_SEED:-83948}

echo "spt scaling  D=$D L=$L K=$K BASE_SEED=$BASE_SEED node=$(hostname)" \
     "MAX_OUTER=${MAX_OUTER:-default} NAN_JUMP_SCALE=${NAN_JUMP_SCALE:-default} EXP=${EXP:-scaling_L12}"
echo "writing results to $RESULTS_ROOT/{data,figures,logs}"

for rep in $(seq 0 $((K - 1))); do
  SEED=$((BASE_SEED + rep)) D=$D RANDOM_INIT=1 STOP_FULL=1 \
    python -u /home/szhou3/diffphase/diff_phase/spt/scripts/main_active_phase.py \
    > "$RESULTS_ROOT/logs/d${D}_L${L}_s$((BASE_SEED + rep)).log" 2>&1 &
done
wait
echo "all $K reps for D=$D finished"
