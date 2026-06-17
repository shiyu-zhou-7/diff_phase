"""
Autonomy benchmark for the cluster-chain backend (task section 5).

Launches K random start points in the t cube (normalized onto the sphere
S^{d-1}, so the phase diagram is a cell decomposition of the sphere), runs the
existing discovery loop from each, and scores everything against the ANALYTIC
ground truth (analytic.cluster_exact.winding -- never ED). Reports three things:

  (i)   distinct phases visited vs cumulative solver queries (curve + array),
  (ii)  detected boundary-crossing locations vs the analytic roots, including the
        predicted closing momentum k* (table),
  (iii) total forward-equivalent cost vs the analytic grid count (the FAIR scan is
        over d-1 sphere dimensions, ~ (1/eps)^(d-1), not the full cube) and a
        random-search baseline scored on analytic labels (table).

Usage (from spt/scripts/):
    python benchmark.py
    D=3 K=4 L=10 EPOCHS=800 MINI_EPOCHS=400 INIT_N=150 BOOTSTRAP_N=80 MAX_OUTER=3 python benchmark.py
    D=8 K=16 L=10 python benchmark.py          # headline (expensive)

Writes ../data/spt_benchmark_*.{npz,json} and ../figures/spt_benchmark_*_phases.png.
"""
import sys, os, json
from dataclasses import replace, asdict

import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig, ClusterConfig, ActiveConfig
from hamiltonians.cluster import build_cluster_model
from workflows.active_phase_discovery import run_active_phase_discovery
from analytic.cluster_exact import winding, boundary_crossings, is_gapless
from training.dataset import normalize_rows
from utils.io import save_pickle, timestamp


def _envint(name, default):
    return int(os.environ[name]) if name in os.environ else default


# ---------------------------------------------------------------------------
# Baselines scored on analytic labels (no ED -- winding is the grader)
# ---------------------------------------------------------------------------
def random_search_baseline(d, n_samples, rng):
    """Sample n_samples points on S^{d-1}, label by winding, return the cumulative
    distinct-phase count and the number of samples needed to see all realized
    phases (a fair random-search competitor on analytic labels)."""
    pts = normalize_rows(np.asarray(jax.random.normal(rng, (n_samples, d))))
    seen, curve = set(), []
    first_all = None
    universe = set()
    # the realized phases on the sphere are 0..d-1 generically
    for i, t in enumerate(pts):
        if not is_gapless(t):
            seen.add(int(winding(t)))
        curve.append(len(seen))
    return {'curve': curve, 'distinct': sorted(seen), 'n_samples': n_samples}


def analytic_grid_count(d, eps):
    """Number of points in a fair epsilon-resolution scan of the sphere S^{d-1}
    (~ (1/eps)^(d-1)), and the naive full-cube count (~ (1/eps)^d) for reference."""
    sphere = int(round((1.0 / eps) ** (d - 1)))
    cube = int(round((1.0 / eps) ** d))
    return {'eps': eps, 'sphere_scan': sphere, 'cube_scan': cube}


# ---------------------------------------------------------------------------
# Run K starts
# ---------------------------------------------------------------------------
def run_benchmark(d, K, L, ae_cfg, cluster_cfg, act_cfg, seed=0):
    key = jax.random.PRNGKey(seed)
    model = build_cluster_model(L=L, d=d, bc=cluster_cfg.bc,
                                sector=cluster_cfg.sector, kappa=cluster_cfg.kappa)

    key, sk = jax.random.split(key)
    starts = normalize_rows(np.asarray(jax.random.normal(sk, (K, d))))

    runs = []
    global_seen = set()
    cum_queries = 0
    pvq = []                     # (cumulative_queries, cumulative_distinct_phases)
    all_crossings = []

    for j, t0 in enumerate(starts):
        key, rk = jax.random.split(key)
        print(f'\n===== start {j+1}/{K}  t0={np.round(t0,3)}  omega0={winding(t0)} =====')
        res = run_active_phase_discovery(
            t_init=t0, model=model,
            ae_cfg=ae_cfg, cluster_cfg=cluster_cfg, act_cfg=act_cfg,
            rng_key=rk, checkpoint_path=None,
        )
        runs.append(res)

        # (i) merge this run's per-step omega/query trail into the global curve
        omegas = res['history']['omega_per_step']
        qrel = res['history']['query_per_step']        # cumulative-within-run
        base = cum_queries
        for w, q in zip(omegas, qrel):
            global_seen.add(int(w))
            pvq.append((base + q, len(global_seen)))
        cum_queries = base + (qrel[-1] if qrel else 0)

        # (ii) analytic boundary crossings along this trajectory
        trail = np.asarray(res['history']['t_per_step'], dtype=float)
        if len(trail) >= 2:
            for c in boundary_crossings(trail):
                all_crossings.append({'start': j, **c})

    pvq = np.array(pvq) if pvq else np.zeros((0, 2))
    return {
        'starts': starts.tolist(),
        'runs_distinct': [r['distinct_phases'] for r in runs],
        'runs_queries': [r['total_queries'] for r in runs],
        'global_distinct': sorted(global_seen),
        'phases_vs_queries': pvq,
        'crossings': all_crossings,
        'total_queries': int(cum_queries),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    d = _envint('D', 3)
    K = _envint('K', 4)
    L = _envint('L', 10)
    eps = float(os.environ.get('EPS', 0.1))

    cluster_cfg = ClusterConfig(L=L, d=d, bc=os.environ.get('BC', 'pbc'),
                                kappa=float(os.environ.get('KAPPA', 0.0)),
                                max_steps_block=_envint('MAX_STEPS', ClusterConfig().max_steps_block))
    ae_cfg = AEConfig()
    act_cfg = ActiveConfig()
    if 'EPOCHS' in os.environ:
        ae_cfg = replace(ae_cfg, epochs=_envint('EPOCHS', ae_cfg.epochs))
    if 'MINI_EPOCHS' in os.environ:
        ae_cfg = replace(ae_cfg, mini_epochs=_envint('MINI_EPOCHS', ae_cfg.mini_epochs))
    if 'MAX_OUTER' in os.environ:
        act_cfg = replace(act_cfg, max_outer_iters=_envint('MAX_OUTER', act_cfg.max_outer_iters))
    if 'INIT_N' in os.environ:
        act_cfg = replace(act_cfg, num_samples_init=_envint('INIT_N', act_cfg.num_samples_init))
    if 'BOOTSTRAP_N' in os.environ:
        act_cfg = replace(act_cfg, num_samples_bootstrap=_envint('BOOTSTRAP_N', act_cfg.num_samples_bootstrap))

    out = run_benchmark(d, K, L, ae_cfg, cluster_cfg, act_cfg,
                        seed=_envint('SEED', 0))

    # baselines
    key = jax.random.PRNGKey(_envint('SEED', 0) + 999)
    rand = random_search_baseline(d, max(out['total_queries'], 200), key)
    grid = analytic_grid_count(d, eps)

    os.makedirs('../data', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)
    ts = timestamp()
    tag = f'd{d}_K{K}_L{L}_{cluster_cfg.bc}'

    # (iii) cost comparison
    cost = {
        'discovery_total_queries': out['total_queries'],
        'discovery_distinct_phases': out['global_distinct'],
        'random_baseline_distinct': rand['distinct'],
        'random_baseline_samples': rand['n_samples'],
        'analytic_grid': grid,
    }

    summary = {
        'config': {'d': d, 'K': K, 'L': L, 'bc': cluster_cfg.bc, 'eps': eps},
        'runs_distinct': out['runs_distinct'],
        'runs_queries': out['runs_queries'],
        'global_distinct': out['global_distinct'],
        'n_crossings_detected': len(out['crossings']),
        'crossings_by_type': {
            tt: sum(1 for c in out['crossings'] if c['type'] == tt)
            for tt in ('z=+1', 'z=-1', 'complex')
        },
        'cost': cost,
    }

    json_path = f'../data/spt_benchmark_{tag}_{ts}.json'
    npz_path = f'../data/spt_benchmark_{tag}_{ts}.npz'
    with open(json_path, 'w') as f:
        json.dump({**summary, 'crossings': out['crossings'],
                   'starts': out['starts']}, f, indent=2)
    np.savez(npz_path, phases_vs_queries=out['phases_vs_queries'],
             random_curve=np.array(rand['curve']))

    # (i) plot: distinct phases vs cumulative queries (discovery vs random baseline)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        if len(out['phases_vs_queries']):
            ax.step(out['phases_vs_queries'][:, 0], out['phases_vs_queries'][:, 1],
                    where='post', label='active discovery')
        ax.step(np.arange(len(rand['curve'])), rand['curve'], where='post',
                label='random search (analytic labels)', alpha=0.7)
        ax.axhline(d, ls='--', color='k', lw=0.8, label=f'all {d} phases')
        ax.set_xlabel('cumulative solver queries')
        ax.set_ylabel('distinct phases visited')
        ax.set_title(f'cluster chain d={d}, L={L}, K={K} starts')
        ax.legend()
        fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.12)
        fig_path = f'../figures/spt_benchmark_{tag}_{ts}_phases.png'
        fig.savefig(fig_path, dpi=150)
        print(f'\nfigure -> {fig_path}')
    except Exception as e:
        print(f'(plot skipped: {e})')

    print('\n=== BENCHMARK SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print(f'\njson -> {json_path}\nnpz  -> {npz_path}')
