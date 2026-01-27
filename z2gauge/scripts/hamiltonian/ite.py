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

