#!/bin/bash
# Scaling experiment: cost of active phase discovery at L=12 (ED) vs d.
#
# Launches the existing workflow (scripts/main_active_phase.py, unchanged,
# default hyperparameters) for d = 4, 6, 8, 10, 12 in parallel.
#
# Run from anywhere:  bash spt/scaling/run_scaling.sh
# Working dir is scaling/runs/ so the workflow's relative '../data' output
# path lands in scaling/data/.  Logs go to scaling/logs/.
set -u
cd "$(dirname "$0")"
mkdir -p data logs runs
cd runs

PY=../../../.venv/bin/python
MAIN=../../scripts/main_active_phase.py

for D in 4 6 8 10 12; do
  echo "launching d=$D L=12"
  D=$D L=12 $PY $MAIN > ../logs/d${D}_L12.log 2>&1 &
done
wait
echo "all runs finished"
