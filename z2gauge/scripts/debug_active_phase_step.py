"""
Debug script for a single step in active phase discovery.
Prints extensive debug information to identify where NaN occurs.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import config

# Enable 64-bit precision
config.update("jax_enable_x64", True)

print("=" * 80)
print("DEBUG: SINGLE STEP ACTIVE PHASE DISCOVERY")
print("=" * 80)

# System parameters
Lx = 2
Ly = 2
h_init = -1.5
j_a = -1.0

print(f"\n[1] System parameters:")
print(f"    Lx = {Lx}, Ly = {Ly}")
print(f"    h_init = {h_init}")
print(f"    j_a = {j_a}")
print(f"    N = 2 * Lx * Ly = {2 * Lx * Ly}")
print(f"    Hilbert space dimension = 2^{2*Lx*Ly} = {2**(2*Lx*Ly)}")

# Build Hamiltonian operators
print(f"\n[2] Building Hamiltonian operators...")
from hamiltonian.z2ham import hamiltonian, sum_star_operators, transverse_field

star_ops = sum_star_operators(Lx, Ly)
trans_ops = transverse_field(Lx, Ly)
print(f"    star_ops shape: {star_ops.shape}")
print(f"    trans_ops shape: {trans_ops.shape}")
print(f"    star_ops finite: {jnp.all(jnp.isfinite(star_ops))}")
print(f"    trans_ops finite: {jnp.all(jnp.isfinite(trans_ops))}")

# Initialize autoencoder parameters
print(f"\n[3] Initializing autoencoder parameters...")
from models.autoencoder import init_params as init_ae
from configs.config import AEConfig

ae_cfg = AEConfig()
D = 2**(2*Lx*Ly)
layers = [D, 512, ae_cfg.latent_dim, 512, D]
print(f"    Layers: {layers}")
print(f"    Latent dimension: {ae_cfg.latent_dim}")

key = jax.random.PRNGKey(ae_cfg.seed)
key, sub = jax.random.split(key)
init_ae_params = init_ae(layers, sub)
print(f"    AE params initialized")
print(f"    AE params type: {type(init_ae_params)}")

# Check AE params structure
def check_params_finite(params, prefix="", indent="    "):
    """Recursively check if parameters are finite."""
    if isinstance(params, dict):
        for k, v in params.items():
            check_params_finite(v, f"{prefix}.{k}", indent)
    elif isinstance(params, (list, tuple)):
        for i, p in enumerate(params):
            check_params_finite(p, f"{prefix}[{i}]", indent)
    else:
        if hasattr(params, 'shape'):
            is_finite = jnp.all(jnp.isfinite(params))
            print(f"{indent}{prefix}: shape={params.shape}, finite={is_finite}")
            if not is_finite:
                print(f"{indent}  ⚠️  NaN/Inf detected in {prefix}!")
                nan_count = jnp.sum(jnp.isnan(params))
                inf_count = jnp.sum(jnp.isinf(params))
                print(f"{indent}  NaN count: {nan_count}, Inf count: {inf_count}")

print(f"\n[4] Checking AE parameters:")
check_params_finite(init_ae_params, "ae_params")

# Initialize ferro_centroid (latent target)
print(f"\n[5] Initializing ferro_centroid (latent target)...")
key, sub = jax.random.split(key)
ferro_centroid = jax.random.normal(sub, (ae_cfg.latent_dim,))
print(f"    ferro_centroid shape: {ferro_centroid.shape}")
print(f"    ferro_centroid finite: {jnp.all(jnp.isfinite(ferro_centroid))}")
print(f"    ferro_centroid sample: {ferro_centroid[:5]}")

# Initialize Hamiltonian parameter
print(f"\n[6] Initializing Hamiltonian parameter...")
ham_param = jnp.array([h_init], dtype=jnp.float64)
print(f"    ham_param: {ham_param}")
print(f"    ham_param finite: {jnp.all(jnp.isfinite(ham_param))}")

# Build Hamiltonian
print(f"\n[7] Building Hamiltonian matrix...")
H = hamiltonian(j_a, ham_param[0], star_ops, trans_ops)
print(f"    H shape: {H.shape}")
print(f"    H finite: {jnp.all(jnp.isfinite(H))}")
if not jnp.all(jnp.isfinite(H)):
    print(f"    ⚠️  NaN/Inf in H!")
    nan_count = jnp.sum(jnp.isnan(H))
    inf_count = jnp.sum(jnp.isinf(H))
    print(f"    NaN count: {nan_count}, Inf count: {inf_count}")

# Add regularization and diagonalize
print(f"\n[8] Adding regularization and diagonalizing...")
reg_strength = 1e-10
diag_indices = jnp.arange(H.shape[0], dtype=H.dtype)
H_reg = H + reg_strength * jnp.diag(diag_indices)
print(f"    H_reg shape: {H_reg.shape}")
print(f"    H_reg finite: {jnp.all(jnp.isfinite(H_reg))}")

print(f"    Computing eigenvalues and eigenvectors...")
eigenvalues, eigenvectors = jnp.linalg.eigh(H_reg)
print(f"    eigenvalues shape: {eigenvalues.shape}")
print(f"    eigenvectors shape: {eigenvectors.shape}")
print(f"    eigenvalues finite: {jnp.all(jnp.isfinite(eigenvalues))}")
print(f"    eigenvectors finite: {jnp.all(jnp.isfinite(eigenvectors))}")

if not jnp.all(jnp.isfinite(eigenvalues)):
    print(f"    ⚠️  NaN/Inf in eigenvalues!")
    nan_count = jnp.sum(jnp.isnan(eigenvalues))
    inf_count = jnp.sum(jnp.isinf(eigenvalues))
    print(f"    NaN count: {nan_count}, Inf count: {inf_count}")
    print(f"    First 10 eigenvalues: {eigenvalues[:10]}")

if not jnp.all(jnp.isfinite(eigenvectors)):
    print(f"    ⚠️  NaN/Inf in eigenvectors!")
    nan_count = jnp.sum(jnp.isnan(eigenvectors))
    inf_count = jnp.sum(jnp.isinf(eigenvectors))
    print(f"    NaN count: {nan_count}, Inf count: {inf_count}")

# Extract ground state
print(f"\n[9] Extracting ground state...")
ground_state = eigenvectors[:, 0]
print(f"    ground_state shape: {ground_state.shape}")
print(f"    ground_state finite: {jnp.all(jnp.isfinite(ground_state))}")

ground_state = jnp.real(ground_state)
print(f"    After taking real part:")
print(f"    ground_state finite: {jnp.all(jnp.isfinite(ground_state))}")
print(f"    ground_state norm: {jnp.linalg.norm(ground_state)}")

# Add batch dimension
print(f"\n[10] Adding batch dimension...")
ground_state = ground_state[None, :]
print(f"    ground_state shape after batch: {ground_state.shape}")
print(f"    ground_state finite: {jnp.all(jnp.isfinite(ground_state))}")

# Fetch latent representation
print(f"\n[11] Fetching latent representation from autoencoder...")
from models.autoencoder import fetch_latent

print(f"    Calling fetch_latent...")
print(f"    Input ground_state shape: {ground_state.shape}")
print(f"    AE params type: {type(init_ae_params)}")

try:
    z = fetch_latent(init_ae_params, ground_state, jax.random.PRNGKey(0))
    print(f"    ✓ fetch_latent completed")
    print(f"    z shape: {z.shape}")
    print(f"    z finite: {jnp.all(jnp.isfinite(z))}")
    if not jnp.all(jnp.isfinite(z)):
        print(f"    ⚠️  NaN/Inf in z!")
        nan_count = jnp.sum(jnp.isnan(z))
        inf_count = jnp.sum(jnp.isinf(z))
        print(f"    NaN count: {nan_count}, Inf count: {inf_count}")
        print(f"    z sample: {z[0, :5]}")
except Exception as e:
    print(f"    ⚠️  ERROR in fetch_latent: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    z = None

if z is not None:
    # Compute loss
    print(f"\n[12] Computing loss...")
    diff = z[0] - ferro_centroid
    print(f"    diff shape: {diff.shape}")
    print(f"    diff finite: {jnp.all(jnp.isfinite(diff))}")
    
    diff_sq = diff**2
    print(f"    diff_sq finite: {jnp.all(jnp.isfinite(diff_sq))}")
    
    diff_sq_sum = jnp.sum(diff_sq)
    print(f"    diff_sq_sum: {diff_sq_sum}")
    print(f"    diff_sq_sum finite: {jnp.isfinite(diff_sq_sum)}")
    
    sqrt_arg = diff_sq_sum + 1e-12
    print(f"    sqrt_arg: {sqrt_arg}")
    print(f"    sqrt_arg finite: {jnp.isfinite(sqrt_arg)}")
    
    sqrt_val = jnp.sqrt(sqrt_arg)
    print(f"    sqrt_val: {sqrt_val}")
    print(f"    sqrt_val finite: {jnp.isfinite(sqrt_val)}")
    
    loss = -sqrt_val / z.shape[-1]
    print(f"    loss: {loss}")
    print(f"    loss finite: {jnp.isfinite(loss)}")
    
    if not jnp.isfinite(loss):
        print(f"    ⚠️  NaN/Inf in loss!")

# Create a detailed debug version of ham_loss for backpropagation tracking
print(f"\n" + "=" * 80)
print("[13] Creating debug version of ham_loss for backpropagation tracking...")
print("=" * 80)

def debug_ham_loss(ham_param, star_ops, trans_ops, ae_params, latent_target):
    """Non-JIT version of ham_loss with extensive debugging."""
    print(f"\n    [DEBUG LOSS] ========== FORWARD PASS ==========")
    
    h = ham_param[0]
    # Check if we're in a traced context (backpropagation)
    is_traced = isinstance(h, jax.core.Tracer)
    if not is_traced:
        print(f"    [DEBUG LOSS] h = {h}")
        print(f"    [DEBUG LOSS] h finite: {jnp.isfinite(h)}")
    else:
        print(f"    [DEBUG LOSS] h = <Traced> (backpropagation active)")
    
    print(f"    [DEBUG LOSS] Building Hamiltonian...")
    H = hamiltonian(-1.0, h, star_ops, trans_ops)
    H_finite = jnp.all(jnp.isfinite(H))
    print(f"    [DEBUG LOSS] H shape: {H.shape}, finite: {H_finite}")
    if not is_traced:
        print(f"    [DEBUG LOSS] H min/max: {float(jnp.min(H)):.6e} / {float(jnp.max(H)):.6e}")
    
    print(f"    [DEBUG LOSS] Adding regularization...")
    reg_strength = 1e-10
    diag_indices = jnp.arange(H.shape[0], dtype=H.dtype)
    H_reg = H + reg_strength * jnp.diag(diag_indices)
    H_reg_finite = jnp.all(jnp.isfinite(H_reg))
    print(f"    [DEBUG LOSS] H_reg finite: {H_reg_finite}")
    if not is_traced:
        print(f"    [DEBUG LOSS] H_reg min/max: {float(jnp.min(H_reg)):.6e} / {float(jnp.max(H_reg)):.6e}")
    
    print(f"    [DEBUG LOSS] Diagonalizing (this is where backprop issues often occur)...")
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_reg)
    evals_finite = jnp.all(jnp.isfinite(eigenvalues))
    evecs_finite = jnp.all(jnp.isfinite(eigenvectors))
    print(f"    [DEBUG LOSS] eigenvalues shape: {eigenvalues.shape}")
    print(f"    [DEBUG LOSS] eigenvalues finite: {evals_finite}")
    print(f"    [DEBUG LOSS] eigenvectors shape: {eigenvectors.shape}")
    print(f"    [DEBUG LOSS] eigenvectors finite: {evecs_finite}")
    
    # Check for degeneracies that might cause gradient issues
    print(f"    [DEBUG LOSS] Checking for near-degeneracies...")
    if len(eigenvalues) > 1 and not is_traced:
        gap_01 = float(eigenvalues[1] - eigenvalues[0])
        gap_12 = float(eigenvalues[2] - eigenvalues[1]) if len(eigenvalues) > 2 else float('inf')
        e0 = float(eigenvalues[0])
        e1 = float(eigenvalues[1])
        print(f"    [DEBUG LOSS] E0 = {e0:.10e}")
        print(f"    [DEBUG LOSS] E1 = {e1:.10e}")
        print(f"    [DEBUG LOSS] Gap E1-E0 = {gap_01:.10e}")
        print(f"    [DEBUG LOSS] Gap E2-E1 = {gap_12:.10e}")
        if gap_01 < 1e-8:
            print(f"    [DEBUG LOSS] ⚠️  VERY SMALL GAP E1-E0! This can cause gradient issues!")
        if gap_12 < 1e-8:
            print(f"    [DEBUG LOSS] ⚠️  VERY SMALL GAP E2-E1! This can cause gradient issues!")
            print(f"    [DEBUG LOSS] ⚠️  DEGENERACY DETECTED! Gradient through eigh() may be NaN!")
    
    print(f"    [DEBUG LOSS] Extracting ground state...")
    ground_state = eigenvectors[:, 0]
    print(f"    [DEBUG LOSS] ground_state shape: {ground_state.shape}")
    print(f"    [DEBUG LOSS] ground_state finite: {jnp.all(jnp.isfinite(ground_state))}")
    print(f"    [DEBUG LOSS] ground_state norm: {jnp.linalg.norm(ground_state):.10e}")
    
    ground_state = jnp.real(ground_state)
    print(f"    [DEBUG LOSS] After real: ground_state finite: {jnp.all(jnp.isfinite(ground_state))}")
    print(f"    [DEBUG LOSS] After real: ground_state norm: {jnp.linalg.norm(ground_state):.10e}")
    
    ground_state = ground_state[None, :]
    print(f"    [DEBUG LOSS] After batch dim: ground_state shape = {ground_state.shape}")
    
    print(f"    [DEBUG LOSS] Calling fetch_latent (autoencoder forward pass)...")
    z = fetch_latent(ae_params, ground_state, jax.random.PRNGKey(0))
    z_finite = jnp.all(jnp.isfinite(z))
    print(f"    [DEBUG LOSS] z shape: {z.shape}, finite: {z_finite}")
    if not z_finite:
        print(f"    [DEBUG LOSS] ⚠️  NaN/Inf in z!")
        if not is_traced:
            nan_count = jnp.sum(jnp.isnan(z))
            inf_count = jnp.sum(jnp.isinf(z))
            print(f"    [DEBUG LOSS] NaN count: {nan_count}, Inf count: {inf_count}")
    elif not is_traced:
        print(f"    [DEBUG LOSS] z min/max: {float(jnp.min(z)):.6e} / {float(jnp.max(z)):.6e}")
    
    print(f"    [DEBUG LOSS] Computing diff...")
    diff = z[0] - latent_target
    diff_finite = jnp.all(jnp.isfinite(diff))
    print(f"    [DEBUG LOSS] diff shape: {diff.shape}")
    print(f"    [DEBUG LOSS] diff finite: {diff_finite}")
    if diff_finite and not is_traced:
        print(f"    [DEBUG LOSS] diff min/max: {float(jnp.min(diff)):.6e} / {float(jnp.max(diff)):.6e}")
    
    print(f"    [DEBUG LOSS] Computing diff**2...")
    diff_sq = diff**2
    print(f"    [DEBUG LOSS] diff_sq finite: {jnp.all(jnp.isfinite(diff_sq))}")
    
    print(f"    [DEBUG LOSS] Computing sum...")
    diff_sq_sum = jnp.sum(diff_sq)
    diff_sq_sum_finite = jnp.isfinite(diff_sq_sum)
    if not is_traced:
        print(f"    [DEBUG LOSS] diff_sq_sum = {float(diff_sq_sum):.10e}, finite: {diff_sq_sum_finite}")
    else:
        print(f"    [DEBUG LOSS] diff_sq_sum finite: {diff_sq_sum_finite}")
    
    print(f"    [DEBUG LOSS] Computing sqrt argument...")
    sqrt_arg = diff_sq_sum + 1e-12
    sqrt_arg_finite = jnp.isfinite(sqrt_arg)
    if not is_traced:
        print(f"    [DEBUG LOSS] sqrt_arg = {float(sqrt_arg):.10e}, finite: {sqrt_arg_finite}")
    else:
        print(f"    [DEBUG LOSS] sqrt_arg finite: {sqrt_arg_finite}")
    
    print(f"    [DEBUG LOSS] Computing sqrt...")
    sqrt_val = jnp.sqrt(sqrt_arg)
    sqrt_val_finite = jnp.isfinite(sqrt_val)
    if not is_traced:
        print(f"    [DEBUG LOSS] sqrt_val = {float(sqrt_val):.10e}, finite: {sqrt_val_finite}")
    else:
        print(f"    [DEBUG LOSS] sqrt_val finite: {sqrt_val_finite}")
    
    print(f"    [DEBUG LOSS] Computing final loss...")
    loss = -sqrt_val / z.shape[-1]
    loss_finite = jnp.isfinite(loss)
    if not is_traced:
        print(f"    [DEBUG LOSS] loss = {float(loss):.10e}, finite: {loss_finite}")
    else:
        print(f"    [DEBUG LOSS] loss finite: {loss_finite}")
    print(f"    [DEBUG LOSS] ========== FORWARD PASS COMPLETE ==========")
    
    return loss

# Test forward pass first
print(f"\n[14] Testing forward pass with debug function...")
try:
    loss_val = debug_ham_loss(ham_param, star_ops, trans_ops, init_ae_params, ferro_centroid)
    print(f"    Forward pass completed. Loss: {loss_val}")
except Exception as e:
    print(f"    ⚠️  ERROR in forward pass: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    loss_val = None

# Now test gradient computation with detailed tracking
print(f"\n" + "=" * 80)
print("[15] Testing gradient computation (BACKPROPAGATION)...")
print("=" * 80)
print("    NOTE: Backpropagation through eigh() can be numerically unstable")
print("    especially when eigenvalues are degenerate or very close together.")
print("=" * 80)

from jax import value_and_grad

# Create a wrapper that prints gradient information
def debug_grad_fn(ham_param, star_ops, trans_ops, ae_params, latent_target):
    """Compute loss and gradient with detailed backprop debugging."""
    print(f"\n    [DEBUG GRAD] ========== BACKPROPAGATION START ==========")
    print(f"    [DEBUG GRAD] Input ham_param: {ham_param}")
    print(f"    [DEBUG GRAD] Input ham_param finite: {jnp.all(jnp.isfinite(ham_param))}")
    
    # Use value_and_grad to get both loss and gradient
    def loss_fn(p):
        return debug_ham_loss(p, star_ops, trans_ops, ae_params, latent_target)
    
    print(f"\n    [DEBUG GRAD] Calling value_and_grad (this triggers backpropagation)...")
    print(f"    [DEBUG GRAD] Backprop will go through:")
    print(f"    [DEBUG GRAD]   1. Loss computation")
    print(f"    [DEBUG GRAD]   2. Autoencoder (fetch_latent)")
    print(f"    [DEBUG GRAD]   3. Eigenvalue/eigenvector extraction (eigh) <- often problematic")
    print(f"    [DEBUG GRAD]   4. Hamiltonian construction")
    print(f"    [DEBUG GRAD]   5. Parameter h")
    
    try:
        val, grads = value_and_grad(loss_fn)(ham_param)
        
        print(f"\n    [DEBUG GRAD] ========== BACKPROPAGATION COMPLETE ==========")
        print(f"    [DEBUG GRAD] Loss value: {val:.10e}")
        print(f"    [DEBUG GRAD] Loss finite: {jnp.isfinite(val)}")
        print(f"    [DEBUG GRAD] Gradient type: {type(grads)}")
        print(f"    [DEBUG GRAD] Gradient shape: {grads.shape if hasattr(grads, 'shape') else 'N/A'}")
        print(f"    [DEBUG GRAD] Gradient finite: {jnp.all(jnp.isfinite(grads))}")
        
        if jnp.all(jnp.isfinite(grads)):
            print(f"    [DEBUG GRAD] Gradient values: {grads}")
            print(f"    [DEBUG GRAD] Gradient norm: {jnp.linalg.norm(grads):.10e}")
            print(f"    [DEBUG GRAD] Gradient min/max: {jnp.min(grads):.10e} / {jnp.max(grads):.10e}")
        else:
            print(f"    [DEBUG GRAD] ⚠️  NaN/Inf in gradients!")
            nan_count = jnp.sum(jnp.isnan(grads))
            inf_count = jnp.sum(jnp.isinf(grads))
            print(f"    [DEBUG GRAD] NaN count: {nan_count}, Inf count: {inf_count}")
            print(f"    [DEBUG GRAD] Gradient values: {grads}")
            print(f"    [DEBUG GRAD] This indicates a numerical issue in backpropagation!")
            print(f"    [DEBUG GRAD] Common causes:")
            print(f"    [DEBUG GRAD]   - Degenerate or near-degenerate eigenvalues")
            print(f"    [DEBUG GRAD]   - Numerical instability in eigh() gradient")
            print(f"    [DEBUG GRAD]   - Very small spectral gap")
            print(f"    [DEBUG GRAD]   - Large values in intermediate computations")
        
        return val, grads
        
    except Exception as e:
        print(f"\n    [DEBUG GRAD] ⚠️  EXCEPTION during backpropagation!")
        print(f"    [DEBUG GRAD] Exception type: {type(e).__name__}")
        print(f"    [DEBUG GRAD] Exception message: {e}")
        import traceback
        traceback.print_exc()
        raise

try:
    print(f"\n    Executing gradient computation...")
    loss_val, grads = debug_grad_fn(ham_param, star_ops, trans_ops, init_ae_params, ferro_centroid)
    
    print(f"\n[16] Gradient computation completed!")
    print(f"    Loss: {loss_val}")
    print(f"    Gradients: {grads}")
    
except Exception as e:
    print(f"\n    ⚠️  ERROR in gradient computation: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Now test the full step
print(f"\n" + "=" * 80)
print("[17] Testing full Hamiltonian step with optimizer...")
print("=" * 80)

from training.ham_optimize import make_ham_step
from configs.config import HamConfig

ham_cfg = HamConfig()
print(f"    Learning rate: {ham_cfg.lr}")
print(f"    Max steps block: {ham_cfg.max_steps_block}")

ham_opt = optax.adam(learning_rate=ham_cfg.lr)
ham_state = ham_opt.init(ham_param)
print(f"    Optimizer state initialized")

ham_step = make_ham_step(ham_opt)
print(f"    ham_step function created")

print(f"\n[18] Calling ham_step (this will use JIT-compiled version)...")
print(f"    Input ham_param: {ham_param}")
print(f"    Input ham_param finite: {jnp.all(jnp.isfinite(ham_param))}")

try:
    print(f"\n    Executing ham_step...")
    ham_param_new, ham_state_new, L, gnorm = ham_step(
        ham_param, star_ops, trans_ops, init_ae_params, ferro_centroid, ham_state
    )
    
    print(f"\n[19] Step completed!")
    print(f"    New ham_param: {ham_param_new}")
    print(f"    ham_param_new finite: {jnp.all(jnp.isfinite(ham_param_new))}")
    print(f"    Loss L: {L}")
    print(f"    Loss L finite: {jnp.isfinite(L)}")
    print(f"    Gradient norm: {gnorm}")
    print(f"    Gradient norm finite: {jnp.isfinite(gnorm)}")
    
    if not jnp.all(jnp.isfinite(ham_param_new)):
        print(f"    ⚠️  NaN/Inf in ham_param_new!")
    if not jnp.isfinite(L):
        print(f"    ⚠️  NaN/Inf in loss L!")
    if not jnp.isfinite(gnorm):
        print(f"    ⚠️  NaN/Inf in gradient norm!")
        
except Exception as e:
    print(f"\n    ⚠️  ERROR in ham_step: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)

