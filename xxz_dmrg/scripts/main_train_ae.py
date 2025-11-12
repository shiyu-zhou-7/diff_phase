import jax
import jax.numpy as jnp
import numpy as np
import pickle

from jax import config
config.update("jax_enable_x64", True)

# GPU configuration via a single default_device context
from contextlib import nullcontext
_gpu = next((d for d in jax.devices() if d.platform == 'gpu'), None)
_devctx = jax.default_device(_gpu) if _gpu is not None else nullcontext()
if _gpu is not None:
    print(f"Using GPU: {_gpu}")
else:
    print("No GPU found, using CPU")

from models.autoencoder import init_params, autoencoder, fetch_latent, cal_ave_m
from training.ae_train import train_autoencoder
from utils.io import save_pickle
from configs.config import AEConfig, ActiveConfig
import matplotlib.pyplot as plt

with _devctx:
    # ----- load ferro dataset -----
    # with open('../data/data_obsZX_dmrg_L=20_ver1.pkl', 'rb') as f:
    # with open('../data/dmrg_data_xxz_dmrg_processed_delta-2.5TO-1.5_h0.0TO0.5_L20_Paulizxy.pkl', 'rb') as f:
    with open('../data/dmrg_data_xxz_dmrg1site_delta-2.5TO-1.5_h0.0TO0.5_L20_n1000_20251111_165934.pkl', 'rb') as f:
    # with open('../data/dmrg_data_xxz_dmrg_delta-2.0TO-1.5_h-3.5TO-1.5_L20.pkl', 'rb') as f:
    # with open('../mps/dmrg_data_xxz_dmrg_delta-2.0TO-1.5_h-3.5TO-1.5_L20.pkl', 'rb') as f:
        data = np.array(pickle.load(f))
    np.random.shuffle(data)

    # simple train/test split 
    ratio = 0.3
    test_size = int(len(data) * ratio)
    idx = np.random.choice(len(data), test_size, replace=False)
    train = np.delete(data, idx)
    X_train = np.array([d['obs'][:60].real for d in train])
    X_test  = np.array([d['obs'][:60].real for d in data[idx]])

    D = X_train.shape[1]
    print(f'Train size: {X_train.shape}, Test size: {X_test.shape}, D = {D}')

    # ----- build & train -----
    ae_cfg = AEConfig()
    act_cfg = ActiveConfig()
    key = jax.random.PRNGKey(ae_cfg.seed)
    layers = [D, 20, 5, 20, D]
    key, sub = jax.random.split(key)
    params = init_params(layers, sub)
    print(f'Autoencoder layers: {layers}')

    params = train_autoencoder(params, X_train,
                               epochs=ae_cfg.epochs,
                               lr=ae_cfg.lr,
                               weight_decay=ae_cfg.weight_decay,
                               drop_p=ae_cfg.dropout_p,
                               center_coeff=act_cfg.center_coeff,
                               seed=ae_cfg.seed,
                               X_test=X_test)

    Z = fetch_latent(params, X_train, jax.random.PRNGKey(0))
    ferro_centroid = jnp.mean(Z, axis=0)
    print('Ferro centroid (latent space): ', ferro_centroid)

    X_rc = autoencoder(params, X_test, 0, key)   ## testing mode, set drop_p = 0
    m_test, m_rc = cal_ave_m(X_test[:, 2*D//3:], D//3), cal_ave_m(X_rc[:, 2*D//3:], D//3)

    plt.figure()
    fs = 15
    plt.plot(range(len(m_test)), m_test, 'o', color='blue', label=f'<m>')
    plt.plot(range(len(m_rc)), m_rc, 'o', color='orange', label=f'<m>_rec')
    plt.legend(fontsize=fs-2)
    plt.xlabel('delta', fontsize=fs)
    plt.ylabel('<O>', fontsize=fs)
    plt.savefig(f'../figures/xxzhdmrgAutoEncoder_ferro_reconstructed_dmrg1site.pdf', bbox_inches='tight')

    save_pickle({'params': params, 'centroid': np.array(ferro_centroid)}, '../models/xxzhdmrg1site_autoencoder_params_latent5_delta-2.5TO-1.5_h0.0TO0.5_L20.pkl')
    print('Saved AE params to ../models/xxzhdmrg1site_autoencoder_params_latent5_delta-2.5TO-1.5_h0.0TO0.5_L20.pkl')