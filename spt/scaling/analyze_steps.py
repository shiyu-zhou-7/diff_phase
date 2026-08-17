"""
Cost accounting for the L=12 scaling experiment (spt/scaling).

Reads every non-checkpoint bundle spt_active_*.pkl in a data dir and reports,
per (d, seed): total simulation steps (= solver queries: one ED solve per inner
step + every dataset sample), decomposed as

    total = inner_steps + init_samples + retrain_samples
    retrain_samples = sum over retrains k of num_samples_bootstrap * (1 + k)
                      (history-aware retrain: 200 per ACCUMULATED center)

plus phases visited and whether coverage {0..d-1} completed (the runs use
STOP_FULL=1, so covered runs stop right at coverage). Aggregates per d.

Usage:
    python analyze_steps.py <data_dir> [--json out.json]
"""
import glob
import json
import os
import pickle
import sys

import numpy as np


def winding(t, tol=1e-6):
    """Inline copy of analytic.cluster_exact.winding (alpha=0 PLUS convention:
    ttilde = (-t_0, t_1, ...)), so this script runs standalone anywhere."""
    t = np.asarray(t, dtype=float).copy()
    t[0] = -t[0]
    coeffs = t[::-1]
    nz = np.nonzero(np.abs(coeffs) > 1e-14)[0]
    if nz.size == 0:
        return 0
    coeffs = coeffs[nz[0]:]
    if coeffs.shape[0] <= 1:
        return 0
    roots = np.roots(coeffs)
    radii = np.abs(roots)
    return int(np.count_nonzero((radii < 1.0) & ~(np.abs(radii - 1.0) < tol)))


def load_runs(data_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(data_dir, 'spt_active_*.pkl'))):
        if path.endswith('_ckpt.pkl'):
            continue
        with open(path, 'rb') as f:
            b = pickle.load(f)
        cfg = b['cfg_snapshot']
        d = cfg['cluster_cfg']['d']
        seed = cfg['cluster_cfg']['seed']
        act = cfg['act_cfg']
        n_steps = len(b['history']['omega_per_step'])
        n_retrains = len(b['bootstrap_history']) - 1
        init_n = act['num_samples_init']
        boot_n = act['num_samples_bootstrap']
        # history-aware retrain: retrain k (1-indexed) trains on 1+k centers
        retrain_samples = sum(boot_n * (1 + k) for k in range(1, n_retrains + 1))
        total = b['total_queries']
        resid = total - (n_steps + init_n + retrain_samples)
        # coverage includes the START phase (the workflow's seen_omegas seeds
        # with winding(t_init); omega_per_step records post-step values only)
        omega_init = winding(cfg['t_init'])
        visited = sorted({omega_init} | set(int(w) for w in b['history']['omega_per_step']))
        covered = set(visited) >= set(range(d))
        runs.append(dict(
            d=d, seed=seed, file=os.path.basename(path),
            total=total, inner_steps=n_steps, init_samples=init_n,
            retrain_samples=retrain_samples, n_retrains=n_retrains,
            residual=resid,     # 0 if the decomposition is exact
            visited=visited, covered=bool(covered),
        ))
    return runs


def report(runs):
    ds = sorted(set(r['d'] for r in runs))
    print(f"{'d':>3} {'cov':>5} {'total: mean+-std':>18} {'min':>6} {'max':>6} "
          f"{'steps':>7} {'init':>5} {'retrain':>8} {'#retr':>6}")
    for d in ds:
        rs = [r for r in runs if r['d'] == d]
        cov = sum(r['covered'] for r in rs)
        covered = [r for r in rs if r['covered']]
        if not covered:
            print(f"{d:>3} {cov:>3}/{len(rs)}   (no covered runs)")
            continue
        tot = np.array([r['total'] for r in covered])
        st = np.array([r['inner_steps'] for r in covered])
        rt = np.array([r['retrain_samples'] for r in covered])
        nr = np.array([r['n_retrains'] for r in covered])
        print(f"{d:>3} {cov:>3}/{len(rs)} {tot.mean():>10.0f}+-{tot.std():<6.0f} "
              f"{tot.min():>6} {tot.max():>6} {st.mean():>7.0f} {rs[0]['init_samples']:>5} "
              f"{rt.mean():>8.0f} {nr.mean():>6.1f}")
    bad = [r for r in runs if r['residual'] != 0]
    if bad:
        print(f"\nWARNING: {len(bad)} runs with nonzero decomposition residual:")
        for r in bad:
            print(f"  {r['file']}: residual={r['residual']}")


if __name__ == '__main__':
    data_dir = sys.argv[1]
    runs = load_runs(data_dir)
    report(runs)
    if '--json' in sys.argv:
        out = sys.argv[sys.argv.index('--json') + 1]
        with open(out, 'w') as f:
            json.dump(runs, f, indent=1)
        print(f'\nper-run records -> {out}')
