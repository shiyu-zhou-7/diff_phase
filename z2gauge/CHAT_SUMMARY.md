# Chat Summary: NaN Gradient Issue Resolution

## Date
November 25, 2024

## Problem
The active phase discovery optimization was encountering NaN gradients during Hamiltonian parameter optimization, causing the optimization to fail with warnings like:
```
[WARNING] NaN/Inf in params/loss/grad. Reverting and jumping past the region.
```

## Root Cause
The NaN gradients were caused by numerical instability when differentiating through `jnp.linalg.eigh()` when eigenvalues were degenerate or very close together. The gradient of the ground state eigenvector becomes ill-defined in these cases.

## Solutions Implemented

### 1. Regularization to Break Degeneracies
**File:** `z2gauge/scripts/training/ham_optimize.py`

Added a small diagonal perturbation to the Hamiltonian before eigendecomposition:
```python
reg_strength = 1e-10
diag_indices = jnp.arange(H.shape[0], dtype=H.dtype)
H_reg = H + reg_strength * jnp.diag(diag_indices)
eigenvalues, eigenvectors = jnp.linalg.eigh(H_reg)
```

**Impact:** Makes eigenvalues distinct, stabilizing gradients through `eigh()`. The perturbation is tiny (`1e-10`) and doesn't affect physics.

### 2. Safe Gradient Handling
**File:** `z2gauge/scripts/training/ham_optimize.py`

Replaced NaN/Inf gradients with zeros to skip updates:
```python
grads_safe = jax.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
```

**Impact:** Prevents NaN from propagating through the optimization.

### 3. Gradient Clipping
**File:** `z2gauge/scripts/training/ham_optimize.py`

Clipped gradients to maximum norm of 10.0:
```python
max_grad_norm = 10.0
grads_clipped = jax.tree_map(lambda g: g * jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-12)), grads_safe)
```

**Impact:** Prevents gradient explosion.

### 4. Parameter Clipping
**File:** `z2gauge/scripts/training/ham_optimize.py`

Clipped Hamiltonian parameters to reasonable range:
```python
ham_param = jnp.clip(ham_param, -5.0, 5.0)
```

**Impact:** Keeps parameters in a physically reasonable range.

### 5. Numerical Stability
**File:** `z2gauge/scripts/training/ham_optimize.py`

Added epsilon to square root in loss function:
```python
loss = -jnp.sqrt(jnp.sum(diff**2) + 1e-12) / z.shape[-1]
```

**Impact:** Prevents numerical issues when `diff` is very small.

### 6. Debugging Infrastructure
**Files:** 
- `z2gauge/scripts/training/ham_optimize.py`
- `z2gauge/scripts/workflows/active_phase_discovery.py`

Added:
- `jax.debug.print()` statements for tracing-compatible debugging
- Eigenvalue gap monitoring
- NaN detection checks in outer loop
- Initial loss computation function

## Results

### Before Fix
- NaN gradients appearing intermittently (every few steps)
- Optimization failing and reverting parameters
- Loss values were finite, but gradients became NaN

### After Fix
- ✅ No NaN gradients observed in test run
- ✅ All gradients remained finite throughout
- ✅ Eigenvalue gaps healthy (~2.48-2.49)
- ✅ Optimization completed successfully through all 6 outer iterations
- ✅ Parameter `h` evolved smoothly from `-1.5` to `-1.5047`

## Files Modified

1. **`z2gauge/scripts/training/ham_optimize.py`**
   - Added regularization to `ham_loss()`
   - Added safe gradient handling in `make_ham_step()`
   - Added gradient and parameter clipping
   - Added debugging statements
   - Added `compute_initial_loss()` function
   - Temporarily disabled `@jit` decorators for debugging

2. **`z2gauge/scripts/workflows/active_phase_discovery.py`**
   - Added NaN detection checks
   - Added call to `compute_initial_loss()`

## Key Technical Details

### Why Regularization Works
When eigenvalues are degenerate, the eigenvector is not unique, making the gradient through `eigh()` ill-defined. By adding a small diagonal perturbation, we ensure all eigenvalues are distinct, making the eigenvector unique and the gradient well-defined.

### Gradient Computation Through Eigendecomposition
JAX's automatic differentiation through `jnp.linalg.eigh()` uses the formula:
```
∂v/∂H = (H - λI)⁺ (∂H/∂θ) v
```
where `(H - λI)⁺` is the pseudo-inverse. When eigenvalues are degenerate, `(H - λI)` is singular, making the gradient undefined.

### Regularization Strength
The regularization strength of `1e-10` was chosen to be:
- Small enough to not affect physics (eigenvalue shifts are negligible)
- Large enough to break numerical degeneracies (larger than floating-point precision)

## Testing

The fix was tested with:
- `Lx = 2, Ly = 1` (4 qubits, 16-dimensional Hilbert space)
- Initial `h = -1.5`
- Learning rate `1e-4`
- `max_steps_block = 10` (reduced for testing)
- 6 outer iterations

All runs completed successfully without NaN errors.

## Recommendations

1. **Re-enable JIT compilation** once debugging is complete (uncomment `@jit` decorators)
2. **Monitor eigenvalue gaps** in production runs to detect potential issues
3. **Consider adaptive regularization** if degeneracies persist in larger systems
4. **Keep debugging infrastructure** for future troubleshooting

## Final Parameter Value

From the test run, the final Hamiltonian parameter value was approximately:
- `h_final ≈ -1.5047`

The exact value is stored in the checkpoint file:
- `../models/active_phase_discovery_checkpoint_lr0.0001_20251125_214623.pkl`
- `../models/active_phase_ham_params_history_h-1.5_2x1_20251125_214623.pkl`

