"""
Train an autoencoder on TFIM ground-state wavefunctions.

Reads:
    ../data/data_ssb_debiased.pkl   (training, ssb phase)
    ../data/data_para.pkl           (eval set, paramagnetic phase)

Writes (with timestamp suffix <ts>):
    ../models/ae_params_<ts>.pkl              # trained AE params + run config
    ../data/latent_eval_<ts>.pkl              # latent-space snapshots over training
    ../models/ae_latent_centroids_<ts>.pkl    # mean latent of ssb / para test sets
    ../figures/ae_train_loss_<ts>.pdf
    ../figures/ae_latent_scatter_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python main_train_ae.py
"""
import sys, os
import numpy as np
from dataclasses import replace, asdict

import jax
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig
from models.autoencoder import init_params, encoder
from training.ae_train import train_autoencoder
from utils.io import save_pickle, load_pickle, timestamp
from utils.plotting import plot_loss_curve, plot_latent_scatter


# ── tunables ──────────────────────────────────────────────────────────────────
ae_cfg = AEConfig()
# ae_cfg = replace(ae_cfg, epochs=2000, layer_widths=(1024, 500, 20, 500, 1024))

DATA_TRAIN_PATH = '../data/data_ssb_debiased.pkl'
DATA_PARA_PATH  = '../data/data_para.pkl'

TEST_FRAC       = 0.3
SHUFFLE_SEED    = 45297

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'AE config: {asdict(ae_cfg)}')

# ── load data ────────────────────────────────────────────────────────────────
np.random.seed(SHUFFLE_SEED)

data = np.array(load_pickle(DATA_TRAIN_PATH))
np.random.shuffle(data)

test_size = int(len(data) * TEST_FRAC)
test_indices = np.random.choice(len(data), test_size, replace=False)
data_test = data[test_indices]
data_train = np.delete(data, test_indices)

x_train = np.array([d['v'] for d in data_train])
x_test  = np.array([d['v'] for d in data_test])
print(f'train shape: {x_train.shape}, test shape: {x_test.shape}')

data_para = np.array(load_pickle(DATA_PARA_PATH))
eval_size = int(len(data_para) * TEST_FRAC)
eval_indices = np.random.choice(len(data_para), eval_size, replace=False)
x_para_eval = np.array([data_para[i]['v'] for i in eval_indices])
print(f'para-eval shape: {x_para_eval.shape}')

# Sanity: configured first layer must match input dimension.
input_dim = x_train.shape[-1]
if ae_cfg.layer_widths[0] != input_dim or ae_cfg.layer_widths[-1] != input_dim:
    print(f'[warn] layer_widths {ae_cfg.layer_widths} disagree with input dim {input_dim} — '
          f'rebuilding layer_widths with dim {input_dim}.')
    lw = list(ae_cfg.layer_widths)
    lw[0] = input_dim
    lw[-1] = input_dim
    ae_cfg = replace(ae_cfg, layer_widths=tuple(lw))

# ── init AE ───────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(ae_cfg.seed)
key, sub = jax.random.split(key)
params = init_params(list(ae_cfg.layer_widths), sub, scale=ae_cfg.init_scale)
print(f'layers: {ae_cfg.layer_widths}')

# ── train ─────────────────────────────────────────────────────────────────────
hist = train_autoencoder(
    params,
    x_train,
    x_test=x_test,
    x_para_eval=x_para_eval,
    epochs=ae_cfg.epochs,
    lr=ae_cfg.lr,
    drop_p=ae_cfg.dropout_p,
    log_every=ae_cfg.log_every,
    eval_every=ae_cfg.eval_every,
    seed=ae_cfg.seed,
)
trained_params = hist['params']

# ── save artifacts ────────────────────────────────────────────────────────────
params_path  = f'../models/ae_params_{ts}.pkl'
latent_path  = f'../data/latent_eval_{ts}.pkl'
centroid_path = f'../models/ae_latent_centroids_{ts}.pkl'
loss_pdf     = f'../figures/ae_train_loss_{ts}.pdf'
scatter_pdf  = f'../figures/ae_latent_scatter_{ts}.pdf'

save_pickle({'params': trained_params, 'config': asdict(ae_cfg)}, params_path)
save_pickle({
    'loss_list':        hist['loss_list'],          # per-epoch training loss
    'latent_ssb_list':  hist['latent_ssb_list'],
    'latent_para_list': hist['latent_para_list'],
    'val_loss_list':    hist['val_loss_list'],
    'mag_list':         hist['mag_list'],
    'config':           asdict(ae_cfg),
}, latent_path)

# centroids: needed by main_optim_h_ae.py
mid = len(trained_params) // 2
z_ssb_final  = encoder(trained_params[0:mid], x_test, 0.0, key)
z_para_final = encoder(trained_params[0:mid], x_para_eval, 0.0, key)
centroid_ssb  = np.array(np.mean(z_ssb_final, axis=0))
centroid_para = np.array(np.mean(z_para_final, axis=0))
save_pickle({
    'centroid_ssb':  centroid_ssb,
    'centroid_para': centroid_para,
    'config':        asdict(ae_cfg),
    'params_path':   params_path,
}, centroid_path)

# ── plots ─────────────────────────────────────────────────────────────────────
plot_loss_curve(hist['loss_list'], loss_pdf, ylabel='AE loss', logy=True)
plot_latent_scatter(z_ssb_final, z_para_final, scatter_pdf, dims=(0, 1))

print(f'\nSaved:')
for p in (params_path, latent_path, centroid_path, loss_pdf, scatter_pdf):
    print(f'  {p}')
