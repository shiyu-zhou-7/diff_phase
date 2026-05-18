import jax
import jax.numpy as jnp
from .z2ham import hamiltonian


def ite_ground_state(
    H: jnp.ndarray,
    *,
    n_steps: int = 150,
    dt: float = 1e-2,
    key: jax.random.PRNGKey = None,
    v0: jnp.ndarray = None,
    eps: float = 1e-12,
):
    """
    Imaginary-time evolution to approximate the ground state of a dense Hamiltonian H.

    Uses a stabilized first-order step:
        v <- (I - dt*(H - E*I)) v
    where E = <v|H|v> is the Rayleigh quotient at the current iterate,
    followed by normalization each step.

    This is JAX-friendly (jit/grad) and avoids eigen-decomposition.

    Args:
        H: (D, D) Hermitian / real-symmetric Hamiltonian
        n_steps: number of ITE steps
        dt: imaginary-time step size
        key: PRNGKey (optional) for random init
        v0: initial vector (optional). If provided, key is ignored.
        eps: small number to avoid divide-by-zero

    Returns:
        v: (D,) normalized approximate ground state (real dtype if H is real)
        E: scalar Rayleigh quotient at final v
    """
    D = H.shape[0]
    dtype = H.dtype

    if v0 is None:
        if key is None:
            key = jax.random.PRNGKey(0)
        v = jax.random.normal(key, (D,), dtype=dtype)
    else:
        v = jnp.asarray(v0, dtype=dtype)

    # normalize init
    v = v / (jnp.linalg.norm(v) + eps)

    def body(v, _):
        Hv = H @ v
        E = jnp.vdot(v, Hv)  # scalar (complex if needed)
        # stabilized Euler step for exp(-dt H)
        v_new = v - dt * (Hv - E * v)
        v_new = v_new / (jnp.linalg.norm(v_new) + eps)
        return v_new, E

    v, Es = jax.lax.scan(body, v, xs=None, length=n_steps)
    # final energy
    E_final = jnp.real(jnp.vdot(v, H @ v))
    # return real vector if H is real
    v = jnp.real(v)
    v = v / (jnp.linalg.norm(v) + eps)
    return v, E_final


def ite_ground_state_from_params(
    j_a: float,
    h: float,
    star_ops: jnp.ndarray,
    trans_ops: jnp.ndarray,
    *,
    n_steps: int = 150,
    dt: float = 1e-2,
    key: jax.random.PRNGKey = None,
):
    """Convenience wrapper matching your current Z2 Hamiltonian construction."""
    H = hamiltonian(jnp.float32(j_a), jnp.float32(h), star_ops, trans_ops)
    H = H.astype(jnp.float32)
    return ite_ground_state(H, n_steps=n_steps, dt=dt, key=key)


def ite_ground_state_batched(
    j_a: float,
    h_batch: jnp.ndarray,           # (B,)
    star_ops: jnp.ndarray,          # (D, D)
    trans_ops: jnp.ndarray,         # (D, D)
    *,
    n_steps: int = 150,
    dt: float = 1e-2,
    key: jax.random.PRNGKey = None,
    eps: float = 1e-12,
):
    """Batched ITE for B independent (j_a, h_b) pairs sharing star/trans ops.

    Avoids materializing the (B, D, D) batched Hamiltonian by exploiting
        H(h_b) @ v_b = j_a * (S @ v_b) + h_b * (T @ v_b)
    The two `V @ ops` products are dispatched as single GEMMs per scan step,
    with the (D, D) operators cached across all B rows. All B samples share
    the same initial v0 (drawn from `key`), matching the fixed-twin convention
    used by the single-sample path.

    Args:
        j_a: scalar star-operator coupling
        h_batch: (B,) transverse-field values
        star_ops, trans_ops: (D, D) symmetric Hermitian operators
        n_steps: ITE iterations
        dt: imaginary-time step
        key: PRNGKey for the shared initial state (default PRNGKey(0))
        eps: numerical safety in normalization

    Returns:
        V: (B, D) normalized approximate ground states
        E: (B,) final per-sample Rayleigh-quotient energies (real)
    """
    dtype = star_ops.dtype
    D = star_ops.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Same initial v0 for every sample (twin-selection convention).
    v0 = jax.random.normal(key, (D,), dtype=dtype)
    v0 = v0 / (jnp.linalg.norm(v0) + eps)
    B = h_batch.shape[0]
    V = jnp.broadcast_to(v0, (B, D))

    j_a_c = jnp.asarray(j_a, dtype=dtype)
    h_c = h_batch.astype(dtype)        # (B,)

    def body(V, _):
        # star_ops and trans_ops are symmetric: V @ S == (S @ V.T).T row-wise.
        SV = V @ star_ops                           # (B, D)  GEMM
        TV = V @ trans_ops                          # (B, D)  GEMM
        HV = j_a_c * SV + h_c[:, None] * TV         # (B, D)
        E = jnp.sum(V * HV, axis=-1)                # (B,)    per-row Rayleigh quotient
        V_new = V - dt * (HV - E[:, None] * V)      # (B, D)
        norms = jnp.linalg.norm(V_new, axis=-1, keepdims=True)
        return V_new / (norms + eps), E

    V, _ = jax.lax.scan(body, V, xs=None, length=n_steps)

    # Final per-sample energy + cleanup (mirrors single-sample function).
    SV = V @ star_ops
    TV = V @ trans_ops
    HV = j_a_c * SV + h_c[:, None] * TV
    E_final = jnp.real(jnp.sum(V * HV, axis=-1))    # (B,)
    V = jnp.real(V)
    norms = jnp.linalg.norm(V, axis=-1, keepdims=True)
    V = V / (norms + eps)
    return V, E_final

