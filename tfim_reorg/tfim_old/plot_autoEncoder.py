import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import os.path

## load data
size_train = 1000
num_epochs = 10000
architecture = [2**10, 1000, 1, 1000, 2**10] 
hyper_param = '_'.join(str(e) for e in architecture)
with open('./tfim_test/data/tfimAutoEncoder_ssb_nnParams_trainSize%d_%s_epoch%d.pickle'%(size_train, hyper_param, num_epochs), 'rb') as f:
    data = pickle.load(f)
    
## print the data layout
print('data type', type(data))
print('data layout', data.keys())

loss_epoch = data['loss_epoch']
h_list = data['h_list']
latent_prm_h = data['latent_prm_h']
loss_h = data['loss_h']
m_rec_h = data['m_rec_h']
m_ori_h = data['m_ori_h']

print(latent_prm_h.shape)

""" plot """
## create a new directory
name_dir = './tfim_test/figures/tfimAutoEncoder_trainSize%d_ssb_%s_epoch%d'%(size_train, hyper_param, num_epochs)
if (not os.path.isfile(name_dir)):
  os.mkdir(name_dir)

## fontsize
fs = 14

## loss function vs epoch
plt.figure()
plt.plot(loss_epoch, 'o')
plt.ylabel('loss',fontsize=fs)
plt.xlabel('epoch',fontsize=fs)
plt.yscale('log')
plt.savefig(name_dir+'/tfim_ae_train_loss.pdf', bbox_inches='tight')
plt.close()

## latent parameter vs h
# plt.figure()
# plt.plot(h_list, latent_prm_h, 'x', markersize=3)
# plt.ylabel('latent parameter',fontsize=fs)
# plt.xlabel('h',fontsize=fs)
# plt.savefig(name_dir+'/tfim_ae_train_latent_vs_h.pdf', bbox_inches='tight')
# plt.close()

## reconstruction loss vs h
plt.figure()
plt.plot(h_list, loss_h, 'x', markersize=3)
plt.ylabel('loss',fontsize=fs)
plt.xlabel('h',fontsize=fs)
plt.savefig(name_dir+'/tfim_ae_train_loss_vs_h.pdf', bbox_inches='tight')
plt.close()

## magnetization vs h
plt.figure()
plt.plot(h_list, m_rec_h, 'o', label='rec')
plt.plot(h_list, m_ori_h, 'o', label='ori')
plt.legend(fontsize=fs)
plt.ylabel('<m>',fontsize=fs)

plt.xlabel('h',fontsize=fs)
plt.savefig(name_dir+'/tfim_ae_train_m.pdf', bbox_inches='tight')
plt.close()

