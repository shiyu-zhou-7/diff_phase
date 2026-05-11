"""
Train an autoencoder on TFIM paramagnetic-phase wavefunctions only.

Reads:
    ../data/data_para.pkl   — 1000 wavefunctions in the paramagnetic phase

Convention note:
    The h values stored in `data_para.pkl` are positive (h ∈ [+1.001, +2.000])
    under the convention the file was generated with. Under this codebase's
    Hamiltonian convention `H(h) = -Σ ZᵢZⱼ + h Σ Xᵢ` the same wavefunctions
    correspond to NEGATIVE-h paramagnetic ground states. The wavefunction
    vectors stored in the file are correct for the negative-h regime; only
    the stored `h` field's sign is opposite. This script does not rewrite
    those `h` values — it just trains the AE on the wavefunction vectors
    as-is. Use this AE with negative-h optim runs.

Writes (with timestamp suffix <ts>):
    ../models/para_ae_params_<ts>.pkl              # trained params + config
    ../models/para_ae_centroid_<ts>.pkl            # mean of normalized para latents
    ../figures/para_ae_train_loss_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python main_train_para_ae.py

To use this AE with main_optim_h_ae.py:
    AE_PARAMS=../models/para_ae_params_<ts>.pkl \\
    AE_CENTROIDS=../models/para_ae_centroid_<ts>.pkl \\
    python main_optim_h_ae.py
"""
import sys, os
import numpy as np
from dataclasses import replace, asdict

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig
from models.autoencoder import init_params, encoder
from training.ae_train import train_autoencoder
from utils.io import save_pickle, load_pickle, timestamp
from utils.plotting import plot_loss_curve


# ── tunables ──────────────────────────────────────────────────────────────────
ae_cfg = AEConfig()
DATA_PARA_PATH = '../data/data_para.pkl'
TEST_FRAC      = 0.3
SHUFFLE_SEED   = 45297

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'AE config: {asdict(ae_cfg)}')

# ── load para-only data ──────────────────────────────────────────────────────
np.random.seed(SHUFFLE_SEED)
data = np.array(load_pickle(DATA_PARA_PATH))
np.random.shuffle(data)

test_size = int(len(data) * TEST_FRAC)
test_idx  = np.random.choice(len(data), test_size, replace=False)
data_test  = data[test_idx]
data_train = np.delete(data, test_idx)

x_train = np.array([d['v'] for d in data_train])
x_test  = np.array([d['v'] for d in data_test])
print(f'train shape: {x_train.shape}, test shape: {x_test.shape}')
print(f'h range in source pickle (file convention): '
      f'[{min(d["h"] for d in data):.4f}, {max(d["h"] for d in data):.4f}]   '
      f'(== negative-h para under H(h)=-ZZ+hX)')

# Sanity: configured first layer must match input dimension.
input_dim = x_train.shape[-1]
if ae_cfg.layer_widths[0] != input_dim or ae_cfg.layer_widths[-1] != input_dim:
    lw = list(ae_cfg.layer_widths)
    lw[0] = input_dim
    lw[-1] = input_dim
    ae_cfg = replace(ae_cfg, layer_widths=tuple(lw))
    print(f'[warn] rebuilt layer_widths to {ae_cfg.layer_widths}')

# ── init AE ───────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(ae_cfg.seed)
key, sub = jax.random.split(key)
params = init_params(list(ae_cfg.layer_widths), sub, scale=ae_cfg.init_scale)
print(f'layers: {ae_cfg.layer_widths}')

# ── train (no x_para_eval — we only have one phase here) ─────────────────────
hist = train_autoencoder(
    params,
    x_train,
    x_test=x_test,
    epochs=ae_cfg.epochs,
    lr=ae_cfg.lr,
    drop_p=ae_cfg.dropout_p,
    log_every=ae_cfg.log_every,
    eval_every=ae_cfg.eval_every,
    seed=ae_cfg.seed,
)
trained_params = hist['params']

# ── compute para centroid (mean of normalized latents on the para train set) ──
mid = len(trained_params) // 2
key, sub = jax.random.split(key)
z_para = encoder(trained_params[0:mid], jnp.asarray(x_train), 0.0, sub)
centroid_para = np.array(np.mean(z_para, axis=0))
print(f'centroid shape: {centroid_para.shape}')

# ── save artifacts ────────────────────────────────────────────────────────────
params_path  = f'../models/para_ae_params_{ts}.pkl'
centroid_path = f'../models/para_ae_centroid_{ts}.pkl'
loss_pdf     = f'../figures/para_ae_train_loss_{ts}.pdf'

save_pickle({
    'params':    trained_params,
    'config':    asdict(ae_cfg),
    'loss_list': hist['loss_list'],
    'data_path': DATA_PARA_PATH,
    'phase':     'para',
    'note':      'Trained on data_para.pkl wavefunctions, which are para-phase '
                 'ground states for negative-h under H(h)=-ZZ+hX (convention flip).',
}, params_path)

save_pickle({
    'centroid_para': centroid_para,
    'centroid_ssb':  centroid_para,   # alias for back-compat with main_optim_h_ae.py
    'phase':         'para',
    'config':        asdict(ae_cfg),
    'params_path':   params_path,
    'note':          'centroid_ssb is an ALIAS of centroid_para here so this '
                     'file drops in to main_optim_h_ae.py via AE_CENTROIDS env var.',
}, centroid_path)

plot_loss_curve(hist['loss_list'], loss_pdf, ylabel='AE loss', logy=True,
                title='para-only AE training loss')

print('\nSaved:')
for p in (params_path, centroid_path, loss_pdf):
    print(f'  {p}')
