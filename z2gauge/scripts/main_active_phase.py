"""
Run the z2gauge active phase discovery pipeline end-to-end.

Reads configs (env-var overrides supported), runs the workflow with
checkpointing, pickles the result bundle, and tees stdout to a `.log` file.

Usage:
    cd z2gauge/scripts && python main_active_phase.py
    H_INIT=-0.5 LX=2 LY=3 python main_active_phase.py
    H_INIT=-0.1 EPOCHS=200 INIT_N=20 BOOTSTRAP_N=15 MAX_OUTER=1 python main_active_phase.py   # smoke

Env-var overrides:
    H_INIT       ham_cfg.h_init                       (default: HamConfig().h_init)
    SEED         ham_cfg.seed                         (default: HamConfig().seed)
    LX, LY       ham_cfg.Lx, ham_cfg.Ly               (default: 2, 3)
    EPOCHS       ae_cfg.epochs                        (default: AEConfig().epochs)
    MAX_OUTER    active_cfg.max_outer_iters           (default: 5)
    INIT_N       active_cfg.num_samples_init          (default: 200)
    BOOTSTRAP_N  active_cfg.num_samples_bootstrap     (default: 100)

Outputs (timestamped, all in ../data/):
    active_phase_h{h_tag}_<ts>.pkl       # final bundle
    active_phase_h{h_tag}_<ts>_ckpt.pkl  # checkpoint, overwritten each retrain
    active_phase_h{h_tag}_<ts>.log       # full stdout transcript
"""
import sys, os
from dataclasses import replace, asdict

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import HamConfig, AEConfig, ActiveConfig
from hamiltonian.z2ham import sum_star_operators, transverse_field
from workflows.active_phase_discovery import run_active_phase_discovery
from utils.io import save_pickle, timestamp


class _Tee:
    """Tiny stdout splitter: writes go to every stream in `streams`."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, msg):
        for s in self._streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


# ── tunables (env-var overridable) ────────────────────────────────────────────
h_init = float(os.environ.get('H_INIT', HamConfig().h_init))
seed   = int(os.environ.get('SEED', HamConfig().seed))
Lx     = int(os.environ.get('LX', HamConfig().Lx))
Ly     = int(os.environ.get('LY', HamConfig().Ly))

ham_cfg    = HamConfig(h_init=h_init, seed=seed, Lx=Lx, Ly=Ly)
ae_cfg     = AEConfig()
active_cfg = ActiveConfig()

# Rebuild the AE layer widths from the actual lattice (D_in = 2^(2*Lx*Ly)).
N = 2 * Lx * Ly
D_in = 2 ** N
ae_cfg = replace(ae_cfg, layer_widths=(D_in, 500, 10, 500, D_in))

if 'EPOCHS' in os.environ:
    ae_cfg = replace(ae_cfg, epochs=int(os.environ['EPOCHS']))
if 'MAX_OUTER' in os.environ:
    active_cfg = replace(active_cfg, max_outer_iters=int(os.environ['MAX_OUTER']))
if 'INIT_N' in os.environ:
    active_cfg = replace(active_cfg, num_samples_init=int(os.environ['INIT_N']))
if 'BOOTSTRAP_N' in os.environ:
    active_cfg = replace(active_cfg, num_samples_bootstrap=int(os.environ['BOOTSTRAP_N']))

# ── paths ─────────────────────────────────────────────────────────────────────
ts = timestamp()
h_tag = f'{h_init:+.2f}'.replace('+', 'p').replace('-', 'm')
bundle_path = f'../data/active_phase_h{h_tag}_{ts}.pkl'
ckpt_path   = f'../data/active_phase_h{h_tag}_{ts}_ckpt.pkl'
log_path    = f'../data/active_phase_h{h_tag}_{ts}.log'

# ── tee stdout to log file (line-buffered, restored in finally) ───────────────
log_file = open(log_path, 'w', buffering=1)
_orig_stdout = sys.stdout
sys.stdout = _Tee(_orig_stdout, log_file)

try:
    print(f'Timestamp: {ts}')
    print(f'h_init={h_init}, seed={seed}, Lx={Lx}, Ly={Ly} (N={N} qubits, D_in={D_in})')
    print(f'ham_cfg:    {asdict(ham_cfg)}')
    print(f'ae_cfg:     {asdict(ae_cfg)}')
    print(f'active_cfg: {asdict(active_cfg)}')
    print(f'bundle:     {bundle_path}')
    print(f'checkpoint: {ckpt_path}')
    print(f'log:        {log_path}')
    print()

    # ── build operators once (independent of h) ────────────────────────────────
    print('Building star + transverse-field operators...')
    star_ops  = sum_star_operators(Lx, Ly)
    trans_ops = transverse_field(Lx, Ly)
    print(f'  star_ops:  {star_ops.shape}')
    print(f'  trans_ops: {trans_ops.shape}')

    # ── run ───────────────────────────────────────────────────────────────────
    rng_key = jax.random.PRNGKey(ham_cfg.seed)
    result = run_active_phase_discovery(
        h_init=h_init,
        j_a=ham_cfg.J,
        star_ops=star_ops, trans_ops=trans_ops,
        ham_cfg=ham_cfg, ae_cfg=ae_cfg, active_cfg=active_cfg,
        rng_key=rng_key,
        checkpoint_path=ckpt_path,
    )

    # ── save final bundle ─────────────────────────────────────────────────────
    save_pickle(result, bundle_path)

    # ── summary ───────────────────────────────────────────────────────────────
    events = result['history']['event_per_step']
    n_total = len(events)
    n_retrain = events.count('retrain')
    n_nan = events.count('nan_kick')
    print()
    print('=== SUMMARY ===')
    print(f'  h_init             = {h_init:+.4f}')
    print(f'  final_h            = {result["final_h"]:+.4f}')
    print(f'  total Adam steps   = {n_total - n_retrain}')
    print(f'  retrains           = {n_retrain}')
    print(f'  NaN kicks          = {n_nan}')
    print(f'  bootstrap_history  = {[round(h, 4) for h in result["bootstrap_history"]]}')
    print(f'  bundle:            {bundle_path}')
    print(f'  checkpoint:        {ckpt_path}')
    print(f'  log:               {log_path}')

finally:
    sys.stdout = _orig_stdout
    log_file.close()
