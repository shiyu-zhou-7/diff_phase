"""
Run the SPT cluster-chain active phase discovery end-to-end (one start point).

Usage (from spt/scripts/):
    python main_active_phase.py
    D=3 L=10 main_active_phase.py
    D=3 L=10 EPOCHS=500 INIT_N=100 BOOTSTRAP_N=60 MAX_OUTER=2 python main_active_phase.py  # smoke

Env overrides: D, L, BC, KAPPA, SEED, EPOCHS, MINI_EPOCHS, MAX_OUTER, INIT_N, BOOTSTRAP_N.
Outputs (timestamped) go to ../data/.
"""
import sys, os
from dataclasses import replace, asdict

import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig, ClusterConfig, ActiveConfig
from hamiltonians.cluster import build_cluster_model
from workflows.active_phase_discovery import run_active_phase_discovery
from utils.io import save_pickle, timestamp
from plot_trajectory import plot_trajectory


def _envint(name, default):
    return int(os.environ[name]) if name in os.environ else default


cluster_cfg = ClusterConfig(
    L=_envint('L', ClusterConfig().L),
    d=_envint('D', ClusterConfig().d),
    bc=os.environ.get('BC', ClusterConfig().bc),
    kappa=float(os.environ.get('KAPPA', ClusterConfig().kappa)),
    seed=_envint('SEED', ClusterConfig().seed),
)
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

os.makedirs('../data', exist_ok=True)
ts = timestamp()
tag = f'd{cluster_cfg.d}_L{cluster_cfg.L}_{cluster_cfg.bc}'
bundle_path = f'../data/spt_active_{tag}_{ts}.pkl'
ckpt_path = f'../data/spt_active_{tag}_{ts}_ckpt.pkl'

print(f'cluster_cfg: {asdict(cluster_cfg)}')
print(f'ae_cfg:      {asdict(ae_cfg)}')
print(f'act_cfg:     {asdict(act_cfg)}')

model = build_cluster_model(
    L=cluster_cfg.L, d=cluster_cfg.d, bc=cluster_cfg.bc,
    sector=cluster_cfg.sector, kappa=cluster_cfg.kappa,
)

# default start: the trivial (transverse-field) corner e_0
t_init = np.zeros(cluster_cfg.d); t_init[0] = 1.0

rng = jax.random.PRNGKey(cluster_cfg.seed)
result = run_active_phase_discovery(
    t_init=t_init, model=model,
    ae_cfg=ae_cfg, cluster_cfg=cluster_cfg, act_cfg=act_cfg,
    rng_key=rng, checkpoint_path=ckpt_path,
)
save_pickle(result, bundle_path)

# auto-render the trajectory figure (non-fatal if plotting fails)
try:
    fig_path = plot_trajectory(bundle_path)
except Exception as e:
    fig_path = None
    print(f'(trajectory plot skipped: {e})')

print()
print('=== SUMMARY ===')
print(f'  distinct phases visited : {result["distinct_phases"]}')
print(f'  total solver queries    : {result["total_queries"]}')
print(f'  final t                 : {np.round(result["final_t"], 3)}')
print(f'  bundle                  : {bundle_path}')
if fig_path:
    print(f'  trajectory figure       : {fig_path}')
