"""One-time export from trusted local experiment files; NOT needed for replotting.
Run each panel in a separate process to isolate the model-specific imports.
"""
from pathlib import Path
import os, sys, runpy, inspect, json, hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'plot_data'
name = sys.argv[1]
settings = {
 'tfim_parameter': ('tfim_reorg', 'plot_combined_active_phase_trajectory.py', []),
 'z2_parameter': ('z2gauge', 'plot_combined_active_phase_trajectory.py', []),
 'xxz_parameter': ('xxz_dmrg', 'plot_combined_trajectory.py', ['437956']),
 'tfim_latent': ('tfim_reorg', 'analyze_phase_latent_from_data.py', ['../data/active_phase_hm0.30_20260511_155044.pkl']),
 'z2_latent': ('z2gauge', 'overlay_traj_variedseed_phaselatent_style.py', []),
 'xxz_latent': ('xxz_dmrg', 'plot_trajectory_in_latent.py', []),
}
model, script, args = settings[name]
source = ROOT/model/'scripts'/script
os.chdir(source.parent)
sys.path.insert(0,str(source.parent))
sys.argv=[str(source),*args]
os.environ['MPLBACKEND']='Agg'
if name=='tfim_latent':
 os.environ['REUSE_AE']='../data/phase_latent_data_ae_ep1000_20260512_225654.pkl'
if name=='xxz_latent':
 os.environ['DATA_PATH']='../data/all_phases_latent_data_20260518_230716.pkl'
 os.environ['AE_PATH']='../models/xxzhdmrg_autoencoder_params_all_phases_latent15_L20.pkl'
import matplotlib.figure
saved = matplotlib.figure.Figure.savefig
exported = False

def save(self,path,*args,**kwargs):
 global exported
 if name=='tfim_latent' and 'pc12_scatter' not in str(path): return
 ns=inspect.currentframe().f_back.f_globals
 data={}
 meta={'source_script':str(source.relative_to(ROOT)), 'source_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'source_files':[], 'original_output':Path(path).name}
 def take(*names):
  for key in names: data[key]=np.asarray(ns[key])
 if name in ('tfim_parameter','z2_parameter'):
  meta['runs']=[]
  for i,run in enumerate(ns['RUNS']):
   b=ns['load_pickle'](run['path']);n=run['max_steps']
   values=np.asarray([b['cfg_snapshot']['h_init']]+list(b['history']['h_per_step'][:n]))
   data[f'run{i}_values']=values
   data[f'run{i}_step']=np.arange(len(values))
   data[f'run{i}_normalized_step']=np.linspace(0,1,len(values))
   data[f'run{i}_event']=np.asarray(['start']+list(b['history']['event_per_step'][:n]),dtype=str)
   meta['runs'].append({k:run[k] for k in ['color','label','max_steps']})
   meta['source_files'].append(str((source.parent/run['path']).resolve().relative_to(ROOT)))
 elif name=='xxz_parameter':
  take('prev_params','run2_params','nan_kick_origins','ae_spots')
  meta['columns']=['Delta','h']
  meta['source_files']=['xxz_dmrg/models/active_phase_discovery_checkpoint_latest.pkl','xxz_dmrg/models/active_phase_resume_history_run_20260505_235151.pkl']
  from phase_diagram import hc_of_delta,_HC_ANCHORS
  data['boundary_Delta']=np.linspace(-2.5,2.5,600)
  data['boundary_h']=hc_of_delta(data['boundary_Delta'])
  data['boundary_anchors']=_HC_ANCHORS
  data['boundary_h_at_minus1']=np.asarray(hc_of_delta(-1.0))
 elif name=='tfim_latent':
  take('z_sub_pca','traj_pca','var_ratio','pca_components','z_mean','z_sub','traj_latents','traj_h_values','traj_event_tags','idx_sub')
  data['train_pca']=data.pop('z_sub_pca')[:,:2]
  data['train_phase']=np.array(['SSB']*500+['PARA']*500)
  data['trajectory_step']=np.array([ns['real_idx'][k] for k in ns['keep']])
  meta['source_files']=['tfim_reorg/data/phase_latent_data_ae_ep1000_20260512_225654.pkl','tfim_reorg/data/active_phase_hm0.30_20260511_155044.pkl','tfim_reorg/data/data_ssb_debiased.pkl','tfim_reorg/data/data_para.pkl']
 elif name=='z2_latent':
  take('traj_pca','var_ratio','P','mu','Z_all','traj_step_idx','traj_h','z_traj_all')
  data['train_pca']=np.concatenate([ns['Z2_conf'],ns['Z2_dec']])
  data['train_phase']=np.array(['confined']*len(ns['Z2_conf'])+['deconfined']*len(ns['Z2_dec']))
  data['pca_components']=data.pop('P');data['z_mean']=data.pop('mu')
  data['trajectory_step']=data.pop('traj_step_idx')
  data['traj_latents']=data.pop('z_traj_all')[data['trajectory_step']]
  for k in ['ae_path','conf_pkl','dec_pkl','traj_path','bundle_path']:meta['source_files'].append(str((source.parent/ns[k]).resolve().relative_to(ROOT)))
 elif name=='xxz_latent':
  take('P_train','P_traj','pca_components','pca_evr','pca_mean','Z_train','Z_traj','train_phase')
  data['train_pca']=data.pop('P_train');data['traj_pca']=data.pop('P_traj')
  data['var_ratio']=data.pop('pca_evr');data['z_mean']=data.pop('pca_mean')
  data['traj_latents']=data.pop('Z_traj')
  for k in ['DATA_PATH','AE_PATH']:meta['source_files'].append(str((source.parent/ns[k]).resolve().relative_to(ROOT)))
  if 'traj_ham_params' in ns['bundle']:data['traj_ham_params']=np.asarray(ns['bundle']['traj_ham_params'])
 for a in data.values():
  if a.dtype.hasobject: raise ValueError('Object arrays are not portable')
 np.savez_compressed(PACKAGE/'data'/f'{name}.npz',**data)
 meta['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data.items()}
 meta['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in meta['source_files']}
 meta['axes']={'xlim':list(self.axes[0].get_xlim()),'ylim':list(self.axes[0].get_ylim()),'xlabel':self.axes[0].get_xlabel(),'ylabel':self.axes[0].get_ylabel()}
 (PACKAGE/'data'/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
 saved(self,ROOT/'tmp/plot_data_export'/f'{name}.png',dpi=120)
 exported=True
 print('EXPORTED',name,{k:v.shape for k,v in data.items()},flush=True)
matplotlib.figure.Figure.savefig=save
runpy.run_path(str(source),run_name='__main__')
assert exported, 'No figure was exported'
