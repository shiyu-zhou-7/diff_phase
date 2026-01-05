"""
Debug script for active phase discovery workflow with comprehensive NaN tracking.
Specifically focuses on DMRG operations where NaN often occurs.
"""
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from contextlib import nullcontext

from jax import config
config.update("jax_enable_x64", True)

# GPU configuration
_gpu = next((d for d in jax.devices() if d.platform == 'gpu'), None)
_devctx = jax.default_device(_gpu) if _gpu is not None else nullcontext()
if _gpu is not None:
    print(f"Using GPU: {_gpu}")
else:
    print("No GPU found, using CPU")

from workflows.active_phase_discovery import active_phase_discovery
from utils.io import save_pickle, load_pickle
from configs.config import AEConfig, HamConfig, ActiveConfig, DMRGConfig
from dmrg.dmrg import DMRG, run_dmrg, lanczos, H_eff
from dmrg.mps import get_random_MPS
from dmrg.hamiltonians import XXZhX
from training.dmrg_optimize import _latent_loss
from dmrg1.run_xxz import dmrg_E, cal_observables

# Monkey-patch DMRG methods to add NaN checking
original_update_bond = DMRG.update_bond
original_update_lenv = DMRG.update_lenv
original_update_renv = DMRG.update_renv
original_sweep = DMRG.sweep

# Global flag to track if NaN was found
_nan_found = False

def check_nan(x, name, location="", verbose=False):
    """Check for NaN/Inf and print detailed info if found.
    
    Args:
        x: Value to check
        name: Name of the variable
        location: Where in the code this check is happening
        verbose: If True, print even when no NaN is found (for debugging)
    """
    global _nan_found
    
    if isinstance(x, (list, tuple)):
        for i, item in enumerate(x):
            check_nan(item, f"{name}[{i}]", location, verbose)
        return False
    
    try:
        x_array = jnp.asarray(x)
        has_nan = jnp.any(jnp.isnan(x_array))
        has_inf = jnp.any(jnp.isinf(x_array))
        
        if has_nan or has_inf:
            _nan_found = True
            print(f"\n{'='*80}")
            print(f"⚠️  NaN/Inf DETECTED at {location}")
            print(f"   Variable: {name}")
            print(f"   Shape: {x_array.shape}")
            print(f"   Dtype: {x_array.dtype}")
            print(f"   Has NaN: {has_nan}")
            print(f"   Has Inf: {has_inf}")
            if has_nan:
                nan_mask = jnp.isnan(x_array)
                nan_count = jnp.sum(nan_mask)
                print(f"   NaN count: {nan_count} out of {x_array.size} elements")
                if x_array.size < 100:  # Only show indices for small arrays
                    nan_indices = jnp.where(nan_mask)
                    print(f"   NaN indices: {nan_indices}")
            if has_inf:
                inf_mask = jnp.isinf(x_array)
                inf_count = jnp.sum(inf_mask)
                print(f"   Inf count: {inf_count} out of {x_array.size} elements")
                if x_array.size < 100:  # Only show indices for small arrays
                    inf_indices = jnp.where(inf_mask)
                    print(f"   Inf indices: {inf_indices}")
            try:
                finite_vals = x_array[jnp.isfinite(x_array)]
                if len(finite_vals) > 0:
                    print(f"   Min finite value: {jnp.min(finite_vals)}")
                    print(f"   Max finite value: {jnp.max(finite_vals)}")
            except:
                pass
            print(f"{'='*80}\n")
            return True
        elif verbose:
            print(f"  ✓ {name} at {location}: OK (no NaN/Inf)")
    except Exception as e:
        print(f"Error checking {name} at {location}: {e}")
    return False

def debug_update_bond(self, i):
    """Wrapped update_bond with NaN checking."""
    print(f"  [DMRG] update_bond({i})")
    
    # Check inputs BEFORE constructing H_eff
    j = (i + 1) % self.psi.L
    
    # Detailed input checking
    lenv_has_nan = check_nan(self.lenvs[i], f"lenv[{i}]", f"update_bond({i}) - input (BEFORE H_eff)")
    renv_has_nan = check_nan(self.renvs[j], f"renv[{j}]", f"update_bond({i}) - input (BEFORE H_eff)")
    
    if lenv_has_nan or renv_has_nan:
        print(f"  [DMRG] ⚠️  CRITICAL: NaN detected in environments BEFORE H_eff construction!")
        print(f"  [DMRG]     lenv[{i}] has NaN: {lenv_has_nan}")
        print(f"  [DMRG]     renv[{j}] has NaN: {renv_has_nan}")
        print(f"  [DMRG]     This is the ROOT CAUSE - NaN propagated from earlier computation")
    
    check_nan(self.MPO.Ws[i], f"MPO.Ws[{i}]", f"update_bond({i}) - input")
    check_nan(self.MPO.Ws[j], f"MPO.Ws[{j}]", f"update_bond({i}) - input")
    
    # Construct H_eff and check immediately
    h_eff = H_eff(self.lenvs[i], self.renvs[j], self.MPO.Ws[i], self.MPO.Ws[j])
    check_nan(h_eff.lenv, "h_eff.lenv", f"update_bond({i}) - H_eff init")
    check_nan(h_eff.renv, "h_eff.renv", f"update_bond({i}) - H_eff init")
    
    theta = self.psi.get_theta(i).reshape(h_eff.shape[0])
    check_nan(theta, "theta", f"update_bond({i}) - before eigensolve")
    
    # Eigenvalue computation
    if self.lanczos:
        print(f"    [DMRG] Using Lanczos for eigensolve")
        try:
            T, V = lanczos(h_eff.matvec, theta, k=4)
            check_nan(T, "Lanczos_T", f"update_bond({i}) - Lanczos T")
            check_nan(V, "Lanczos_V", f"update_bond({i}) - Lanczos V")
            
            evals, evecs = jnp.linalg.eigh(T)
            check_nan(evals, "evals", f"update_bond({i}) - eigh(evals)")
            check_nan(evecs, "evecs", f"update_bond({i}) - eigh(evecs)")
            
            if V.shape[1] != evecs.shape[0]:
                print(f"    [DMRG] WARNING: Shape mismatch V.shape[1]={V.shape[1]}, evecs.shape[0]={evecs.shape[0]}")
            
            theta_new = V @ evecs[:, jnp.argmin(evals)]
            check_nan(theta_new, "theta_new", f"update_bond({i}) - after Lanczos")
            theta_new = theta_new.reshape(h_eff.theta_shape)
        except Exception as e:
            print(f"    [DMRG] ERROR in Lanczos: {e}")
            raise
    else:
        print(f"    [DMRG] Using dense eigensolve")
        try:
            h_eff_dense = h_eff.to_dense()
            check_nan(h_eff_dense, "h_eff_dense", f"update_bond({i}) - to_dense()")
            
            evals, evecs = jnp.linalg.eigh(h_eff_dense)
            check_nan(evals, "evals", f"update_bond({i}) - eigh(evals)")
            check_nan(evecs, "evecs", f"update_bond({i}) - eigh(evecs)")
            
            theta_new = evecs[:, jnp.argmin(evals)].reshape(h_eff.theta_shape)
            check_nan(theta_new, "theta_new", f"update_bond({i}) - after eigh")
        except Exception as e:
            print(f"    [DMRG] ERROR in dense eigensolve: {e}")
            raise
    
    # SVD and truncation
    from dmrg.mps import split_and_truncate
    A, Sj, B = split_and_truncate(theta_new, h_eff.theta_shape, self.chi_max, self.eps)
    check_nan(A, "A", f"update_bond({i}) - after SVD")
    check_nan(Sj, "Sj", f"update_bond({i}) - after SVD (singular values)")
    check_nan(B, "B", f"update_bond({i}) - after SVD")
    
    # Update MPS
    Si = self.psi.Ss[i]
    check_nan(Si, "Si", f"update_bond({i}) - before update")
    
    Bprev = jnp.tensordot(jnp.diag(1.0 / Si), A, axes=(1, 0))
    check_nan(Bprev, "Bprev", f"update_bond({i}) - after first tensordot")
    
    Bprev = jnp.tensordot(Bprev, jnp.diag(Sj), axes=(2, 0))
    check_nan(Bprev, "Bprev", f"update_bond({i}) - after second tensordot")
    
    self.psi.Ss[j] = Sj
    self.psi.Bs[i] = Bprev
    self.psi.Bs[j] = B
    check_nan(self.psi.Ss[j], f"psi.Ss[{j}]", f"update_bond({i}) - final")
    check_nan(self.psi.Bs[i], f"psi.Bs[{i}]", f"update_bond({i}) - final")
    check_nan(self.psi.Bs[j], f"psi.Bs[{j}]", f"update_bond({i}) - final")
    
    self.update_lenv(i)
    self.update_renv((i + 1) % self.psi.L)

def debug_update_lenv(self, i):
    """Wrapped update_lenv with NaN checking."""
    j = (i + 1) % self.psi.L
    lenv_i = self.lenvs[i]
    W = self.MPO.Ws[i]
    B = self.psi.Bs[i]
    
    check_nan(lenv_i, f"lenv[{i}]", f"update_lenv({i}) - input")
    check_nan(W, f"MPO.Ws[{i}]", f"update_lenv({i}) - input")
    check_nan(B, f"psi.Bs[{i}]", f"update_lenv({i}) - input")
    
    S = jnp.diag(self.psi.Ss[i])
    Sinv = jnp.diag(1.0 / self.psi.Ss[j])
    check_nan(S, f"S[{i}]", f"update_lenv({i})")
    check_nan(Sinv, f"Sinv[{j}]", f"update_lenv({i})")
    
    G = jnp.tensordot(S, B, axes=(1, 0))
    check_nan(G, "G", f"update_lenv({i}) - after first tensordot")
    
    A = jnp.tensordot(G, Sinv, axes=(2, 0))
    check_nan(A, "A", f"update_lenv({i}) - after second tensordot")
    
    A_ = jnp.conj(A)
    lenv_new = jnp.tensordot(lenv_i, A, axes=(0, 0))
    check_nan(lenv_new, "lenv_new", f"update_lenv({i}) - after third tensordot")
    
    lenv_new = jnp.tensordot(lenv_new, W, axes=[[0, 2], [0, 3]])
    check_nan(lenv_new, "lenv_new", f"update_lenv({i}) - after fourth tensordot")
    
    lenv_new = jnp.tensordot(lenv_new, A_, axes=[[0, 3], [0, 1]])
    check_nan(lenv_new, f"lenv[{j}]", f"update_lenv({i}) - final")
    
    self.lenvs[j] = lenv_new

def debug_update_renv(self, i):
    """Wrapped update_renv with NaN checking."""
    print(f"  [DMRG] update_renv({i})")
    
    renv_i = self.renvs[i]
    W = self.MPO.Ws[i]
    B = self.psi.Bs[i]
    
    check_nan(renv_i, f"renv[{i}]", f"update_renv({i}) - input")
    check_nan(W, f"MPO.Ws[{i}]", f"update_renv({i}) - input")
    check_nan(B, f"psi.Bs[{i}]", f"update_renv({i}) - input")
    
    B_ = jnp.conj(B)
    renv_new = jnp.tensordot(B, renv_i, axes=(2, 0))
    check_nan(renv_new, "renv_new", f"update_renv({i}) - after first tensordot")
    
    renv_new = jnp.tensordot(renv_new, W, axes=[[1, 2], [3, 1]])
    check_nan(renv_new, "renv_new", f"update_renv({i}) - after second tensordot")
    
    renv_new = jnp.tensordot(renv_new, B_, axes=[[1, 3], [2, 1]])
    
    target_idx = (i - 1) % self.L
    check_nan(renv_new, f"renv[{target_idx}]", f"update_renv({i}) - final (before assignment)")
    
    # Check if we're about to overwrite a previously valid renv
    if self.renvs[target_idx] is not None:
        old_renv = self.renvs[target_idx]
        has_old_nan = jnp.any(jnp.isnan(jnp.asarray(old_renv)))
        if has_old_nan:
            print(f"  [DMRG] WARNING: Overwriting renv[{target_idx}] that already had NaN")
        else:
            print(f"  [DMRG] Overwriting renv[{target_idx}] (was valid)")
    
    self.renvs[target_idx] = renv_new
    
    # Verify after assignment
    check_nan(self.renvs[target_idx], f"renv[{target_idx}]", f"update_renv({i}) - after assignment")

def debug_sweep(self):
    """Wrapped sweep with NaN checking."""
    print(f"[DMRG] Starting sweep (forward)")
    for i in range(self.psi.num_bonds - 1):
        self.update_bond(i)
    print(f"[DMRG] Starting sweep (backward)")
    for i in range(self.psi.num_bonds - 1, 0, -1):
        self.update_bond(i)

# Wrap DMRG.__init__ to check environments during initialization
original_dmrg_init = DMRG.__init__

def debug_dmrg_init(self, psi, MPO, chi_max, eps=1e-14, lanczos=True):
    """Wrapped DMRG.__init__ with NaN checking."""
    print(f"[DMRG] DMRG.__init__ called: L={psi.L}, chi_max={chi_max}, lanczos={lanczos}")
    
    # Call original init
    original_dmrg_init(self, psi, MPO, chi_max, eps, lanczos)
    
    # Check initial environments
    print(f"[DMRG] Checking initial environments after __init__...")
    for i in range(self.L):
        if self.lenvs[i] is not None:
            check_nan(self.lenvs[i], f"lenv[{i}]", f"DMRG.__init__ - initial")
        if self.renvs[i] is not None:
            check_nan(self.renvs[i], f"renv[{i}]", f"DMRG.__init__ - initial")
    
    # Check after update_renv calls in __init__
    print(f"[DMRG] __init__ calls update_renv for initialization...")
    # The original __init__ calls update_renv for i in range(L-1, 1, -1)
    # We've already wrapped update_renv, so those will be checked automatically

# Monkey-patch the DMRG class
DMRG.__init__ = debug_dmrg_init
DMRG.update_bond = debug_update_bond
DMRG.update_lenv = debug_update_lenv
DMRG.update_renv = debug_update_renv
DMRG.sweep = debug_sweep

# Wrap run_dmrg
original_run_dmrg = run_dmrg

def debug_run_dmrg(L, model, dmrg_cfg):
    """Wrapped run_dmrg with NaN checking."""
    print(f"\n[DMRG] run_dmrg called: L={L}, max_bond={dmrg_cfg.max_bond}, sweeps={dmrg_cfg.sweeps}, lanczos={dmrg_cfg.lanczos_bool}")
    
    # Check model
    check_nan(model.Ws, "model.Ws", "run_dmrg - model initialization")
    
    # Initialize MPS
    psi = get_random_MPS(L, d=2, bond_dim=1)
    check_nan(psi.Ss, "psi.Ss", "run_dmrg - initial MPS")
    check_nan(psi.Bs, "psi.Bs", "run_dmrg - initial MPS")
    
    # Initialize DMRG
    print(f"[DMRG] Initializing DMRG...")
    dmrg = DMRG(psi, model, chi_max=dmrg_cfg.max_bond, lanczos=dmrg_cfg.lanczos_bool)
    
    # Detailed check of all environments after initialization
    print(f"[DMRG] Checking all environments after initialization...")
    for i in range(L):
        if dmrg.lenvs[i] is not None:
            check_nan(dmrg.lenvs[i], f"lenv[{i}]", f"run_dmrg - after DMRG init")
        if dmrg.renvs[i] is not None:
            check_nan(dmrg.renvs[i], f"renv[{i}]", f"run_dmrg - after DMRG init")
    
    # Run sweeps
    for sweep_num in range(dmrg_cfg.sweeps):
        print(f"\n[DMRG] Sweep {sweep_num + 1}/{dmrg_cfg.sweeps}")
        try:
            dmrg.sweep()
            
            # Check all environments after each sweep
            print(f"[DMRG] Checking all environments after sweep {sweep_num + 1}...")
            for i in range(L):
                if dmrg.lenvs[i] is not None:
                    check_nan(dmrg.lenvs[i], f"lenv[{i}]", f"run_dmrg - after sweep {sweep_num + 1}")
                if dmrg.renvs[i] is not None:
                    check_nan(dmrg.renvs[i], f"renv[{i}]", f"run_dmrg - after sweep {sweep_num + 1}")
            
            check_nan(dmrg.psi.Ss, f"psi.Ss", f"run_dmrg - after sweep {sweep_num + 1}")
            check_nan(dmrg.psi.Bs, f"psi.Bs", f"run_dmrg - after sweep {sweep_num + 1}")
        except Exception as e:
            print(f"[DMRG] ERROR during sweep {sweep_num + 1}: {e}")
            # Check environments at error point
            print(f"[DMRG] Checking environments at error point...")
            for i in range(L):
                if dmrg.lenvs[i] is not None:
                    check_nan(dmrg.lenvs[i], f"lenv[{i}]", f"run_dmrg - ERROR point")
                if dmrg.renvs[i] is not None:
                    check_nan(dmrg.renvs[i], f"renv[{i}]", f"run_dmrg - ERROR point")
            raise
    
    print(f"[DMRG] run_dmrg completed successfully")
    return dmrg.psi

# Replace run_dmrg in the module
import dmrg.dmrg as dmrg_module
dmrg_module.run_dmrg = debug_run_dmrg

# Wrap _latent_loss
def debug_latent_loss(delta_h, ae_params, ferro_centroid, observables_list, dmrg_cfg, L):
    """Wrapped _latent_loss with NaN checking."""
    print(f"\n[_latent_loss] Called with delta={delta_h[0]}, h={delta_h[1]}")
    check_nan(delta_h, "delta_h", "_latent_loss - input")
    check_nan(ferro_centroid, "ferro_centroid", "_latent_loss - input")
    
    delta, h = delta_h[0], delta_h[1]
    
    # Create model
    model = XXZhX(L, delta, h)
    check_nan(model.Ws, "model.Ws", "_latent_loss - model creation")
    
    # Run DMRG
    print(f"[_latent_loss] Running DMRG...")
    psi = run_dmrg(L, model, dmrg_cfg)
    check_nan(psi.Ss, "psi.Ss", "_latent_loss - after DMRG")
    check_nan(psi.Bs, "psi.Bs", "_latent_loss - after DMRG")
    
    # Compute observables
    print(f"[_latent_loss] Computing observables...")
    obs_list = psi.get_site_exp_val(observables_list)
    check_nan(obs_list, "obs_list", "_latent_loss - after get_site_exp_val")
    
    obs = jnp.stack(
        [jnp.real(jnp.asarray(x)).reshape(()) for x in obs_list],
        axis=0
    )
    check_nan(obs, "obs", "_latent_loss - after stacking")
    
    # Encode with AE
    print(f"[_latent_loss] Encoding with AE...")
    from models.autoencoder import fetch_latent
    z = fetch_latent(ae_params, obs[None, :], jax.random.PRNGKey(0))[0]
    check_nan(z, "z", "_latent_loss - after AE encoding")
    
    # Compute loss
    diff = z - ferro_centroid
    check_nan(diff, "diff", "_latent_loss - before loss")
    loss = jnp.sum(diff * diff)
    check_nan(loss, "loss", "_latent_loss - final")
    
    # Note: can't use float() here as loss is a JAX tracer during AD
    print(f"[_latent_loss] Loss = {loss}")
    return loss

# Replace _latent_loss in the module
import training.dmrg_optimize as dmrg_opt_module
dmrg_opt_module._latent_loss = debug_latent_loss
# Also update the make_opt_step to use the debug version
original_make_opt_step = dmrg_opt_module.make_opt_step

# Wrap dmrg_E from dmrg1
original_dmrg_E = dmrg_E

def debug_dmrg_E(L, delta, h, conf=1e-4, test=False, chi_max=10, max_sweep=5):
    """Wrapped dmrg_E with NaN checking."""
    print(f"\n[dmrg_E] Called: L={L}, delta={delta}, h={h}, chi_max={chi_max}, max_sweep={max_sweep}")
    check_nan(jnp.array([delta, h]), "[delta, h]", "dmrg_E - input")
    
    from dmrg1.dmrg_xxz import random_MPS, XXZhX_MPO, DMRG, expect_mpo
    
    init_mps = random_MPS(L=L, d=2, chi_max=chi_max)
    check_nan(init_mps.Ms, "init_mps.Ms", "dmrg_E - initial MPS")
    
    mpo = XXZhX_MPO(L=L, d=2, delta=delta, h=h)
    check_nan(mpo.Ws, "mpo.Ws", "dmrg_E - MPO")
    
    dmrg = DMRG(init_mps, mpo, eps=-1., chi_max=chi_max, test=False)
    check_nan(dmrg.LPs, "dmrg.LPs", "dmrg_E - after DMRG init")
    check_nan(dmrg.RPs, "dmrg.RPs", "dmrg_E - after DMRG init")
    
    diff = 5. * conf
    counter = 0
    while diff > conf:
        if test:
            print(f"  [dmrg_E] Sweep {counter}")
        
        e_in = expect_mpo(dmrg.MPS, mpo)
        check_nan(e_in, "e_in", f"dmrg_E - sweep {counter} before")
        
        dmrg.left_to_right()
        check_nan(dmrg.MPS.Ms, "MPS.Ms", f"dmrg_E - sweep {counter} after left_to_right")
        
        dmrg.right_to_left()
        check_nan(dmrg.MPS.Ms, "MPS.Ms", f"dmrg_E - sweep {counter} after right_to_left")
        
        e_end = expect_mpo(dmrg.MPS, mpo)
        check_nan(e_end, "e_end", f"dmrg_E - sweep {counter} after")
        
        diff = abs(e_in - e_end)
        counter += 1
        if counter > max_sweep:
            break
    
    check_nan(dmrg.MPS.Ms, "MPS.Ms", "dmrg_E - final")
    check_nan(dmrg.e[-1], "energy", "dmrg_E - final energy")
    print(f"[dmrg_E] Completed: energy={dmrg.e[-1]}, sweeps={counter}")
    return dmrg.MPS, dmrg.e[-1]

# Replace dmrg_E
import dmrg1.run_xxz as run_xxz_module
run_xxz_module.dmrg_E = debug_dmrg_E
# Also update in workflows module
import workflows.active_phase_discovery as apd_module
apd_module.dmrg_E = debug_dmrg_E

def run_single_step(delta=-1.5, h=-0.3, L=20):
    """Run a single optimization step for debugging."""
    global _nan_found
    _nan_found = False
    
    print(f'\n{"="*80}')
    print(f'Running SINGLE STEP of active phase discovery')
    print(f'Parameters: delta={delta}, h={h}, L={L}')
    print(f'{"="*80}\n')
    
    from workflows.active_phase_discovery import sample_params, generate_data_xxzh
    from training.ae_train import train_autoencoder
    from training.dmrg_optimize import make_opt_step
    from models.autoencoder import fetch_latent, init_params as init_ae
    import optax
    
    with _devctx:
        # Setup observables
        sx = jnp.array([[0, 1], [1, 0]])
        sz = jnp.array([[1, 0], [0, -1]])
        sy = jnp.array([[0, -1j], [1j, 0]])
        
        observables_list = [(i, sz) for i in range(L)] + [(i, sx) for i in range(L)] + [(i, sy) for i in range(L)]
        
        D = len(observables_list)
        print(f'Observable vector dimension D = {D}')
        
        # Setup AE
        key = jax.random.PRNGKey(AEConfig().seed)
        layers = [D, 20, AEConfig().latent_dim, 20, D]
        key, sub = jax.random.split(key)
        ae_params = init_ae(layers, sub)
        print(f'Initialized AE with layers: {layers}')
        
        # Bootstrap centroid
        key, sub = jax.random.split(key)
        deltas, hs = sample_params(delta, h, 1, 0.5, 0.5, sub)
        dmrg_cfg = DMRGConfig()
        dmrg_cfg.max_bond = 1
        dmrg_cfg.sweeps = 1
        dmrg_cfg.lanczos_bool = False
        print(f'DMRG Config: max_bond={dmrg_cfg.max_bond}, sweeps={dmrg_cfg.sweeps}, lanczos={dmrg_cfg.lanczos_bool}')
        
        print(f'Generating bootstrap data...')
        X = generate_data_xxzh(L, deltas, hs, observables_list, dmrg_cfg)
        check_nan(X, "X", "bootstrap data")
        
        print(f'Training AE for bootstrap...')
        ae_params = train_autoencoder(
            ae_params, X, epochs=AEConfig().epochs, lr=AEConfig().lr, 
            weight_decay=AEConfig().weight_decay, drop_p=AEConfig().dropout_p, 
            center_coeff=ActiveConfig().center_coeff, seed=AEConfig().seed
        )
        Z = fetch_latent(ae_params, X, jax.random.PRNGKey(0))
        ferro_centroid = jnp.mean(Z, axis=0)
        check_nan(ferro_centroid, "ferro_centroid", "after bootstrap")
        print(f'Ferro centroid shape: {ferro_centroid.shape}')
        
        # Setup optimizer
        ham_param = jnp.array([delta, h], dtype=jnp.float64)
        opt = optax.adam(learning_rate=HamConfig().lr)
        opt_state = opt.init(ham_param)
        opt_step = make_opt_step(opt, L, dmrg_cfg, observables_list)
        
        print(f'\n{"="*80}')
        print(f'Running SINGLE optimization step...')
        print(f'Initial ham_param: delta={ham_param[0]}, h={ham_param[1]}')
        print(f'{"="*80}\n')
        
        # Run single step
        try:
            ham_param, opt_state, loss_val, gnorm = opt_step(
                ham_param, ae_params, ferro_centroid, opt_state
            )
            
            print(f'\n{"="*80}')
            print(f'Step completed!')
            print(f'  Loss: {float(loss_val):.6e}')
            print(f'  Gradient norm: {float(gnorm):.6e}')
            print(f'  New delta: {float(ham_param[0]):.6f}')
            print(f'  New h: {float(ham_param[1]):.6f}')
            print(f'{"="*80}\n')
            
            check_nan(ham_param, "ham_param", "after step")
            check_nan(loss_val, "loss_val", "after step")
            check_nan(gnorm, "gnorm", "after step")
            
            if _nan_found:
                print("⚠️  WARNING: NaN/Inf values were detected during the step")
            else:
                print("✓ No NaN/Inf values detected during the step")
                
        except Exception as e:
            print(f"\n❌ ERROR during single step: {e}")
            if _nan_found:
                print("⚠️  NaN/Inf values were detected before the error occurred")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main debugging function."""
    global _nan_found
    _nan_found = False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Timestamp: {timestamp}')
    print(f'Starting active phase discovery with comprehensive NaN tracking...\n')
    print(f'This script will track NaN/Inf values throughout the workflow,')
    print(f'with special focus on DMRG operations.\n')
    
    with _devctx:
        # Optionally: load a pretrained AE and centroid
        try:
            file = '../models/xxzhdmrg_autoencoder_params.pkl'
            checkpoint = load_pickle(file)
            init_params = checkpoint['params']
            ferro_centroid = checkpoint['centroid']
            print(f'LOADED CHECKPOINT file: {file}.')
            print(f"Centroid shape: {ferro_centroid.shape}")
            check_nan(init_params, "init_params", "main - loaded checkpoint")
            check_nan(ferro_centroid, "ferro_centroid", "main - loaded checkpoint")
        except Exception as e:
            print(f'NO CHECKPOINT FOUND or ERROR loading: {e}. INITIALIZING FROM SCRATCH.')
            init_params = None
            ferro_centroid = None
        
        try:
            # Explicitly set lanczos to False for debugging
            dmrg_cfg = DMRGConfig()
            dmrg_cfg.max_bond = 1
            dmrg_cfg.sweeps = 1
            dmrg_cfg.lanczos_bool = False
            print(f"DMRG Config: max_bond={dmrg_cfg.max_bond}, sweeps={dmrg_cfg.sweeps}, lanczos={dmrg_cfg.lanczos_bool}")
            
            params, centroid, hist = active_phase_discovery(
                L=20,
                init_delta=-1.5,
                init_h=-0.3,
                ae_cfg=AEConfig(),
                ham_cfg=HamConfig(),
                act_cfg=ActiveConfig(),
                dmrg_cfg=dmrg_cfg,
                init_ae_params=init_params,
                ferro_centroid=ferro_centroid,
                max_outer_iters=6,
            )
            
            print("\n✅ Active phase discovery completed successfully!")
            if _nan_found:
                print("⚠️  WARNING: NaN/Inf values were detected during execution (see logs above)")
            else:
                print("✓ No NaN/Inf values detected during execution")
            save_pickle(
                {'params': params, 'centroid': np.array(centroid), 'hist': hist}, 
                f'../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl'
            )
            print(f'Saved ../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
            
        except Exception as e:
            print(f"\n❌ ERROR in active phase discovery: {e}")
            if _nan_found:
                print("⚠️  NaN/Inf values were detected before the error occurred")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    import sys
    
    # Check if user wants to run single step
    if len(sys.argv) > 1 and sys.argv[1] == "--single-step":
        # Optional: parse delta and h from command line
        delta = -1.5
        h = -0.3
        if len(sys.argv) > 2:
            delta = float(sys.argv[2])
        if len(sys.argv) > 3:
            h = float(sys.argv[3])
        run_single_step(delta=delta, h=h)
    else:
        main()
