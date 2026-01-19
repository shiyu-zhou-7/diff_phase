# Why Degenerate Eigenvalues Cause NaN in Backpropagation Through eigh()

## The Problem

When eigenvalues are degenerate or very close together, the gradient computation through `jnp.linalg.eigh()` becomes numerically unstable and produces NaN.

## Mathematical Explanation

### 1. Eigenvalue Decomposition Gradient Formula

The gradient of eigenvectors with respect to the matrix involves terms like:

```
∂v_i/∂H = Σ_{j≠i} (v_j v_j^T) / (λ_i - λ_j) · ∂H
```

where:
- `v_i` is the i-th eigenvector
- `λ_i` is the i-th eigenvalue  
- `H` is the input matrix

### 2. The Singularity Problem

Notice the term `1/(λ_i - λ_j)` in the formula. When eigenvalues are close or degenerate:

- **If λ_i ≈ λ_j**: The denominator `(λ_i - λ_j)` becomes very small
- **Division by small number**: `1/(very_small_number)` becomes very large or infinite
- **Numerical instability**: Floating point errors amplify, leading to NaN or Inf

### 3. Your Specific Case

From your debug output:
- `Gap E1-E0 = 2.8096369082e+00` ✓ (safe)
- `Gap E2-E1 = 1.7763568394e-15` ✗ (essentially zero!)

When computing gradients:
- Gradient terms involving `1/(E2 - E1)` become `1/(1.77e-15) ≈ 5.6e+14`
- This huge number, combined with floating point errors, leads to NaN
- The gradient computation fails catastrophically

### 4. Why Eigenvectors Are Not Unique at Degeneracies

At degeneracies, there's a **rotation freedom** in the degenerate subspace:
- If E1 = E2, any linear combination of v1 and v2 is also an eigenvector
- The gradient is not well-defined because the eigenvectors are not uniquely determined
- Automatic differentiation tries to compute a gradient that doesn't exist in a well-defined sense

### 5. Why Regularization Helps (But May Not Be Enough)

The regularization `H_reg = H + 1e-10 * diag(indices)` tries to:
- Break exact degeneracies by making eigenvalues slightly different
- However, `1e-10` may be too small if the gap is `~1e-15`
- The gap needs to be larger than numerical precision to be stable

## Solutions

1. **Increase regularization strength**: Make `reg_strength` larger (e.g., `1e-8` or `1e-6`)
2. **Use a different approach**: Project onto the ground state subspace instead of using a single eigenvector
3. **Use finite differences**: For very degenerate cases, approximate gradients numerically
4. **Check for degeneracies**: Skip gradient computation when gaps are too small

## References

- JAX documentation on `eigh()` gradients: The gradient is undefined at degeneracies
- Numerical linear algebra: Eigenvalue problems are ill-conditioned near degeneracies
- Automatic differentiation: Requires smooth, well-defined functions




