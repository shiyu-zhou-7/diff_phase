"""
Full-Hilbert (no parity projection) wavefunction-input AE: can it classify the
8 winding phases of the d=8 cluster chain?

Difference from latent_separability_wavefn.py (which uses the fixed P=+1 SECTOR
vector, 2^(L-1)-dim): here we feed the FULL 2^L wavefunction at L=10 (= 1024-dim),
NOT parity-projected, so the AE architecture is exactly [1024, 512, 128, 512, 1024]
(hidden=512, latent=128). No active-discovery bundle is needed; the t-points and
8-class labels are generated directly from the analytic winding (free-fermion, no ED).

CAVEAT (load-bearing physics): without fixing parity the ground state is DEGENERATE
in the symmetry-broken windings (cat states across the two parity sectors), so eigh
returns an arbitrary vector within the degenerate subspace. Sign gauge-fixing does
NOT remove that subspace arbitrariness. Classification of the broken phases may
therefore look worse than the sector-input run -- that is the degeneracy artifact the
P=+1 projection was designed to avoid, not a failure of the AE.

Metric / plotting helpers are imported from latent_separability.py (numpy-only,
no sklearn, per repo convention).

Usage (from spt/scripts/):
    D=8 L=10 python classify_wavefn_full.py
    D=8 L=10 EPOCHS=6000 TARGET_N=300 REBUILD=1 python classify_wavefn_full.py
env knobs: D(=#couplings d) L BC KAPPA TARGET_N POOL SEED EPOCHS LATENT_DIM HIDDEN
           CENTER_COEFF REBUILD
"""
import sys, os

import numpy as np
import scipy.sparse as sp
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import replace

from configs.config import AEConfig, ClusterConfig
from hamiltonians.cluster import (
    _term_full, _kappa_full, build_cluster_model, gd_solver_ed, lift_to_full,
)
from analytic.cluster_exact import winding, is_gapless
from training.dataset import normalize_rows
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from utils.io import load_pickle, save_pickle, timestamp
# reuse the metric + LDA helpers from the observable-space diagnostics
from latent_separability import (
    knn_accuracy, silhouette, confusion, lda_fit, PHASE_COLORS, _envint,
)


def _envfloat(name, default):
    return float(os.environ[name]) if name in os.environ else default


# ---------------------------------------------------------------------------
# t-points + analytic 8-class labels (free-fermion winding; NO ED)
# ---------------------------------------------------------------------------
def build_labels(d, target_n, pool, seed, sign='free', dominant_match=False):
    """Sample `pool` points on S^{d-1}, label by analytic winding, keep up to
    `target_n` non-gapless points per realized phase. Returns (t (N,d), phase (N,)).

    sign='free' (default): isotropic, both signs. 'neg': force every t_alpha < 0
    (t_alpha = -|gaussian|); 'pos': force every t_alpha > 0. The overall sign of t
    does not change the winding (roots of g unchanged), but restricting the octant
    changes WHICH region of each phase is sampled.

    dominant_match=True: for phase omega keep only points whose DOMINANT (largest
    |t_alpha|) coupling is t_omega -- i.e. winding(t)==omega AND argmax|t|==omega.
    This picks the deep interior of each phase (near the pure point e_omega), the
    cleanest separable representatives, away from boundaries."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((pool, d))
    if sign == 'neg':
        g = -np.abs(g)
    elif sign == 'pos':
        g = np.abs(g)
    elif sign != 'free':
        raise ValueError(f"sign must be 'free', 'neg', or 'pos', got {sign!r}")
    ts = normalize_rows(g)
    gap = np.array([is_gapless(t) for t in ts])
    lab = np.array([(-1 if gp else int(winding(t))) for t, gp in zip(ts, gap)])
    dom = np.argmax(np.abs(ts), axis=1)
    keep_t, keep_phase = [], []
    for w in sorted(set(lab) - {-1}):
        sel = (lab == w)
        if dominant_match:
            sel = sel & (dom == w)        # t_w is the dominant coupling
        idx = np.where(sel)[0][:target_n]
        keep_t.append(ts[idx])
        keep_phase.append(np.full(len(idx), w))
        msg = f'  phase {w}: {len(idx)}'
        if len(idx) < target_n:
            msg += f'/{target_n} (kept all available in pool)'
        print(msg)
    return np.concatenate(keep_t, axis=0), np.concatenate(keep_phase, axis=0)


# ---------------------------------------------------------------------------
# Full-Hilbert (no parity projection) gauge-fixed ground state
# ---------------------------------------------------------------------------
def full_terms(L, d, bc, kappa):
    """Constant full-space sparse term operators (2^L x 2^L), built once."""
    T = [_term_full(L, alpha, bc) for alpha in range(d)]
    Tk = _kappa_full(L, bc) if kappa != 0.0 else None
    return T, Tk


def _sign_fix(psi):
    """Largest-|amplitude| positive, so the same physical state maps to one vector."""
    k = int(np.argmax(np.abs(psi)))
    s = np.sign(psi[k])
    return psi * (s if s != 0 else 1.0)


def gauge_fixed_wavefn_sector_lift(ts, model):
    """Solve the UNIQUE ground state in the fixed P=+1 sector, then lift it back to
    the full 2^L space (zeros on the P=-1 basis states). Returns (N, 2^L).

    Fixing the sector removes the cross-parity ground-state degeneracy of the broken
    phases, so the state is a smooth, unique function of t -- unlike the full-space
    eigh in gauge_fixed_wavefn_full, which returns an arbitrary degenerate-subspace
    vector there. The AE still sees a full 2^L-dim wavefunction."""
    rows = []
    for t in ts:
        _, psi_sec = gd_solver_ed(jnp.asarray(t, dtype=jnp.float64), model)
        psi_full = np.real(np.asarray(lift_to_full(psi_sec, model))).ravel()
        rows.append(_sign_fix(psi_full))
    return np.stack(rows, axis=0)


def gauge_fixed_wavefn_full(ts, T, Tk, kappa):
    """Dense-eigh the FULL 2^L H(t) = sum_a t_a T_a (+ kappa Tk), take the lowest
    eigenvector, sign-gauge-fix (largest-|amplitude| positive). Returns (N, 2^L).

    NB: in symmetry-broken windings the ground state is degenerate, so the lowest
    eigenvector is only defined up to a rotation within that subspace (see caveat
    in the module docstring). Sign-fixing makes it reproducible, not unique."""
    rows = []
    for t in ts:
        H = T[0] * float(t[0])
        for a in range(1, len(T)):
            H = H + T[a] * float(t[a])
        if Tk is not None:
            H = H + kappa * Tk
        Hd = np.asarray(H.todense(), dtype=np.float64)
        Hd = 0.5 * (Hd + Hd.T)
        _, v = np.linalg.eigh(Hd)
        psi = np.real(v[:, 0]).ravel()
        rows.append(_sign_fix(psi))
    return np.stack(rows, axis=0)


def main():
    cc = ClusterConfig()
    d = _envint('D', 8)
    L = _envint('L', 10)
    bc = os.environ.get('BC', cc.bc)
    kappa = _envfloat('KAPPA', cc.kappa)

    ae_cfg = AEConfig()
    ae_cfg = replace(
        ae_cfg,
        hidden=_envint('HIDDEN', 512),
        latent_dim=_envint('LATENT_DIM', 128),
        center_coeff=_envfloat('CENTER_COEFF', 0.0),  # off: classification probe
    )
    # ENC_HIDDEN: comma-separated encoder hidden widths BETWEEN input and latent,
    # e.g. "512,256" -> [1024,512,256,128,256,512,1024]. Default: single HIDDEN width.
    _eh = os.environ.get('ENC_HIDDEN')
    enc_hidden = [int(x) for x in _eh.split(',')] if _eh else [ae_cfg.hidden]
    arch_tag = 'h' + 'x'.join(str(w) for w in enc_hidden)
    target_n = _envint('TARGET_N', 200)
    pool = _envint('POOL', 40000)
    seed = _envint('SEED', 0)
    epochs = _envint('EPOCHS', ae_cfg.epochs)
    rebuild = bool(os.environ.get('REBUILD'))
    # SECTOR=1: solve the UNIQUE P=+1 ground state and lift to the full 2^L space
    # (kills the broken-phase degeneracy); default: arbitrary full-space eigh.
    sector_mode = bool(os.environ.get('SECTOR'))
    kind = 'sectorlift' if sector_mode else 'full'
    t_sign = os.environ.get('T_SIGN', 'free')   # 'free' | 'neg' | 'pos'
    sign_tag = '' if t_sign == 'free' else f'_{t_sign}t'
    dominant_match = bool(os.environ.get('DOMINANT'))  # keep only t_w-dominant points
    dom_tag = '_dom' if dominant_match else ''

    tag = f'd{d}_L{L}_{bc}_n{target_n}{sign_tag}{dom_tag}'
    print(f'{kind} wavefunction classification probe  '
          f'(d={d}, L={L}, bc={bc}, kappa={kappa})')

    # --- dataset: t-points + gauge-fixed full 2^L wavefunctions (cached) ---
    wf_cache = f'../data/spt_wavefn_{kind}_{tag}.pkl'
    if os.path.exists(wf_cache) and not rebuild:
        print(f'loading cached wavefunctions {wf_cache}')
        ds = load_pickle(wf_cache)
        Xwf, phase = np.asarray(ds['X']), np.asarray(ds['phase']).astype(int)
    else:
        print(f'sampling t + analytic labels (target_n={target_n}, pool={pool}, '
              f't_sign={t_sign}, dominant_match={dominant_match})')
        train_t, phase = build_labels(d, target_n, pool, seed, sign=t_sign,
                                      dominant_match=dominant_match)
        phase = phase.astype(int)
        if sector_mode:
            print(f'ED-solving {len(train_t)} UNIQUE P=+1 ground states, '
                  f'lifting to full dim {2**L} ...')
            model = build_cluster_model(L=L, d=d, bc=bc, sector=+1, kappa=kappa)
            Xwf = gauge_fixed_wavefn_sector_lift(train_t, model)
        else:
            print(f'ED-solving {len(train_t)} FULL-space ground states (dim {2**L}) ...')
            T, Tk = full_terms(L, d, bc, kappa)
            Xwf = gauge_fixed_wavefn_full(train_t, T, Tk, kappa)
        save_pickle({'X': Xwf, 't': train_t, 'phase': phase}, wf_cache)
        print(f'  wavefunctions -> {wf_cache}')

    n_phases = len(np.unique(phase))
    Dim = Xwf.shape[1]
    layers = [Dim] + enc_hidden + [ae_cfg.latent_dim] + enc_hidden[::-1] + [Dim]
    print(f'input dim = {Dim}  (= 2^{L}),  N = {len(phase)},  '
          f'{n_phases} phases,  layers = {layers}  (chance = {1/n_phases:.3f})')

    # --- train AE with 1-fidelity loss (cached) ---
    ae_cache = f'../data/spt_ae_wavefn_{kind}_{tag}_z{ae_cfg.latent_dim}_{arch_tag}_fid.pkl'
    if os.path.exists(ae_cache) and not rebuild:
        print(f'loading cached AE {ae_cache}')
        ae_params = load_pickle(ae_cache)['params']
    else:
        print(f'training AE  layers={layers}  epochs={epochs}  '
              f'center_coeff={ae_cfg.center_coeff}  loss=1-fidelity')
        params0 = init_params(layers, jax.random.PRNGKey(ae_cfg.seed), scale=ae_cfg.init_scale)
        ae_params = train_autoencoder(
            params0, jnp.asarray(Xwf), epochs=epochs, lr=ae_cfg.lr,
            weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p,
            center_coeff=ae_cfg.center_coeff, seed=ae_cfg.seed, loss='fidelity',
        )
        save_pickle({'params': ae_params, 'layers': layers, 'tag': tag}, ae_cache)
        print(f'  AE -> {ae_cache}')

    key = jax.random.PRNGKey(0)
    Z = np.asarray(fetch_latent(ae_params, jnp.asarray(Xwf), key))

    # --- separation metrics: latent vs raw-wavefunction baseline ---
    print(f'\n=== separation metrics ({kind} wavefunction input) ===')
    print(f'{"space":16s} {"kNN k=1":>9s} {"kNN k=5":>9s} {"silhouette":>11s}')
    z_acc, z_yte, z_pred = knn_accuracy(Z, phase)
    r_acc, _, _ = knn_accuracy(Xwf, phase)
    print(f'{"latent":16s} {z_acc[1]:9.3f} {z_acc[5]:9.3f} {silhouette(Z, phase):11.3f}')
    print(f'{"raw wavefn":16s} {r_acc[1]:9.3f} {r_acc[5]:9.3f} {silhouette(Xwf, phase):11.3f}')

    print('\n=== confusion matrix (latent, k=5; rows=true, cols=pred) ===')
    M = confusion(z_yte, z_pred, n_phases)
    print('      ' + ' '.join(f'{j:4d}' for j in range(n_phases)))
    for i in range(n_phases):
        print(f'  {i:2d} |' + ' '.join(f'{M[i, j]:4d}' for j in range(n_phases)))

    # --- LDA(2) supervised view (no trajectory: no active run at this config) ---
    W, mu = lda_fit(Z, phase, k=2)
    Z2 = (Z - mu) @ W
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for w in sorted(np.unique(phase)):
        m = phase == w
        ax.scatter(Z2[m, 0], Z2[m, 1], s=10, alpha=0.35,
                   color=PHASE_COLORS.get(int(w), '#333'), linewidths=0,
                   label=f'$\\omega$={w}')
    ax.set_xlabel('LDA1'); ax.set_ylabel('LDA2')
    ax.set_title(f'{kind} wavefunction AE latent, LDA(2)  '
                 f'(d={d}, L={L}, z={ae_cfg.latent_dim})')
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)

    ts = timestamp()
    out = f'../figures/spt_latent_lda_wavefn_{kind}_{tag}_z{ae_cfg.latent_dim}_{ts}.png'
    os.makedirs('../figures', exist_ok=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nLDA figure -> {out}')


if __name__ == '__main__':
    main()
