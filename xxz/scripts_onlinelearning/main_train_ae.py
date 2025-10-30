import jax
import jax.numpy as jnp
import numpy as np
import pickle

from jax import config
config.update("jax_enable_x64", True)

from models.autoencoder import init_params, autoencoder, fetch_latent
from training.ae_train import train_autoencoder
from physics.hamiltonian import cal_m_reconstructed
from utils.io import save_pickle
from configs.config import AEConfig, ActiveConfig
import matplotlib.pyplot as plt

# ----- load ferro dataset -----
with open('../data/data_ferro_xxzh_debiased.pkl', 'rb') as f:
    data = np.array(pickle.load(f))
np.random.shuffle(data)

# simple train/test split 
ratio = 0.3
test_size = int(len(data) * ratio)
idx = np.random.choice(len(data), test_size, replace=False)
train = np.delete(data, idx)
X_train = np.array([d['v'].real for d in train])
X_test  = np.array([d['v'].real for d in data[idx]])

N = int(np.log2(X_train.shape[1]))
D = 2 ** N

# ----- build & train -----
ae_cfg = AEConfig()
act_cfg = ActiveConfig()
key = jax.random.PRNGKey(ae_cfg.seed)
layers = [D, 256, ae_cfg.latent_dim, 256, D]
key, sub = jax.random.split(key)
params = init_params(layers, sub)

params = train_autoencoder(params, X_train,
                           epochs=ae_cfg.epochs,
                           lr=ae_cfg.lr,
                           weight_decay=ae_cfg.weight_decay,
                           drop_p=ae_cfg.dropout_p,
                           center_coeff=act_cfg.center_coeff,
                           seed=ae_cfg.seed)

Z = fetch_latent(params, X_train, jax.random.PRNGKey(0))
ferro_centroid = jnp.mean(Z, axis=0)

X_rc = autoencoder(params, X_test, 0, key)   ## testing mode, set drop_p = 0
m_rc = cal_m_reconstructed(X_rc)
m_test = cal_m_reconstructed(X_test)

plt.figure()
fs = 15
plt.plot(range(len(m_test)), m_test, 'o', color='blue', label=f'<m>')
plt.plot(range(len(m_test)), m_rc, 'o', color='orange', label=f'<m>_rec')
plt.legend(fontsize=fs-2)
plt.xlabel('delta', fontsize=fs)
plt.ylabel('<m>', fontsize=fs)
plt.savefig(f'../figures/xxzhAutoEncoder_ferrodb_reconstructed_magnetization_epoch{ae_cfg.epochs}.pdf', bbox_inches='tight')

save_pickle({'params': params, 'centroid': np.array(ferro_centroid)}, '../models/xxzh_autoencoder_params.pkl')
print('Saved AE params to ../models/xxzh_autoencoder_params.pkl')