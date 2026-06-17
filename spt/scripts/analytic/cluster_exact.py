"""
Analytic ground truth for the generalized cluster (stabilizer) chain.

Model (length-L spin-1/2 chain, open d couplings t = (t_0, ..., t_{d-1})):

    H(t) = - sum_{alpha=0}^{d-1} t_alpha * sum_i  Z_i X_{i+1} ... X_{i+alpha-1} Z_{i+alpha}

    alpha=0 : -t_0 sum_i X_i              (transverse field)
    alpha=1 : -t_1 sum_i Z_i Z_{i+1}      (Ising)
    alpha=2 : -t_2 sum_i Z_i X_{i+1} Z_{i+2}  (cluster)
    general : Z_i (X-string length alpha-1) Z_{i+alpha}

Every term is a product of an even number of Z's, so all commute with the
global Z2 parity P = prod_i X_i. The model is class BDI (real symmetric H), and
the topological invariant is an integer winding number.

Jordan-Wigner maps H to free Majorana fermions with Bloch symbol

    f(k) = sum_alpha t_alpha exp(i*alpha*k),

equivalently the complex polynomial

    g(z) = sum_alpha t_alpha z^alpha      (z = e^{ik} on the unit circle).

g is degree (d-1); it has d-1 roots in C. The number of roots strictly inside
the unit disk is the winding number omega in {0, ..., d-1}; a root ON the unit
circle means the bulk gap closes (a phase boundary). This whole module is pure
numpy and depends on NOTHING in the solver -- it is the grader.

Diagnostics that need the actual ground state (string order, entanglement-
spectrum degeneracy) live with the solver (hamiltonians/observables), because
they are measured, not symbolic. They cross-check `winding` independently of the
root finder (validation section 4.2).

References for a referee:
  - omega = (#zeros of g inside |z|=1) by the argument principle, since
    winding of f(k)=g(e^{ik}) about 0 over k in [0,2pi) counts enclosed zeros of
    the analytic g (no poles).
  - phases: t_0 dominant -> omega=0 (trivial paramagnet); t_1 dominant ->
    omega=1 (Z2 broken); t_2 dominant -> omega=2 (cluster SPT, no local order
    parameter).
"""
import numpy as np


# ----------------------------------------------------------------------------
# Bloch symbol and polynomial roots
# ----------------------------------------------------------------------------
def f_symbol(t, k):
    """Bloch symbol f(k) = sum_alpha t_alpha e^{i alpha k}.

    t : (d,) real coupling vector.
    k : scalar or array of momenta.
    Returns complex array broadcast over k.
    """
    t = np.asarray(t, dtype=float)
    alphas = np.arange(t.shape[0])
    k = np.asarray(k, dtype=float)
    # outer: e^{i * alpha * k} summed against t_alpha
    phases = np.exp(1j * np.multiply.outer(k, alphas))  # (..., d)
    return phases @ t


def g_roots(t, tol_lead=1e-14):
    """Roots of g(z) = sum_alpha t_alpha z^alpha.

    numpy.roots wants the highest-degree coefficient first, so we reverse t.
    Leading (high-degree) zeros are stripped by numpy.roots automatically,
    which correctly lowers the polynomial degree; a pure point e_alpha
    (g = t_alpha z^alpha) therefore returns alpha roots all at z = 0.
    """
    t = np.asarray(t, dtype=float)
    coeffs = t[::-1]  # p[0] z^(d-1) + ... + p[d-1]
    # strip leading (high-degree) ~zero coefficients so numpy.roots gives the
    # true reduced-degree root set rather than spurious huge roots.
    nz = np.nonzero(np.abs(coeffs) > tol_lead)[0]
    if nz.size == 0:
        return np.array([], dtype=complex)  # g identically 0; treat as no roots
    coeffs = coeffs[nz[0]:]
    if coeffs.shape[0] <= 1:
        return np.array([], dtype=complex)  # constant polynomial: no roots
    return np.roots(coeffs)


def winding(t, tol=1e-6):
    """Topological label omega = number of roots of g(z) with |z| < 1.

    Integer in {0, ..., d-1}. Roots within `tol` of the unit circle are treated
    as ON the circle (gapless / boundary) and NOT counted as inside; the caller
    should check `is_gapless` to know the label sits on a boundary. Returns an
    int (the analytic phase label at ANY L, one root-find per call).
    """
    roots = g_roots(t)
    if roots.size == 0:
        return 0
    radii = np.abs(roots)
    on_circle = np.abs(radii - 1.0) < tol
    inside = (radii < 1.0) & ~on_circle
    return int(np.count_nonzero(inside))


def is_gapless(t, tol=1e-6):
    """True if g has a root on the unit circle (bulk gap closes -> boundary)."""
    roots = g_roots(t)
    if roots.size == 0:
        return False
    return bool(np.any(np.abs(np.abs(roots) - 1.0) < tol))


def classify(t, tol=1e-6):
    """Full analytic label for t: omega, gapless flag, and the root set.

    Convenience bundle around `winding` / `is_gapless` for diagnostics & tests.
    """
    roots = g_roots(t)
    radii = np.abs(roots) if roots.size else np.array([])
    on_circle = np.abs(radii - 1.0) < tol if roots.size else np.array([], bool)
    inside = (radii < 1.0) & ~on_circle if roots.size else np.array([], bool)
    return {
        'omega': int(np.count_nonzero(inside)),
        'gapless': bool(np.any(on_circle)),
        'roots': roots,
        'radii': radii,
    }


# ----------------------------------------------------------------------------
# Finite-size gap from the free-fermion BdG spectrum
# ----------------------------------------------------------------------------
def allowed_momenta(L, bc='pbc', sector=+1):
    """Allowed single-particle momenta for the chosen boundary condition/sector.

    PBC fixed-parity:
      sector = +1 (even): NS / antiperiodic set, k in (2pi/L)(Z + 1/2)
      sector = -1 (odd) : R  / periodic    set, k in (2pi/L) Z
    OBC: standing-wave modes k = pi*(m+1)/(L+1), m = 0..L-1. NOTE this plane-wave
      form is exact only for the nearest-neighbour symbol (d <= 2); for d > 2 the
      OBC single-particle problem is not a clean standing wave and `gap` with
      bc='obc' should be cross-checked against a direct BdG diagonalization.
    """
    n = np.arange(L)
    if bc == 'pbc':
        if sector == +1:
            return 2.0 * np.pi * (n + 0.5) / L
        elif sector == -1:
            return 2.0 * np.pi * n / L
        raise ValueError(f'sector must be +1 or -1, got {sector!r}')
    elif bc == 'obc':
        return np.pi * (n + 1.0) / (L + 1.0)
    raise ValueError(f"bc must be 'pbc' or 'obc', got {bc!r}")


def single_particle_energies(t, L, bc='pbc', sector=+1):
    """|f(k)| at the allowed momenta -- the single-particle BdG energies."""
    ks = allowed_momenta(L, bc, sector)
    return np.abs(f_symbol(t, ks))


def gap(t, L, bc='pbc', sector=+1):
    """Closed-form many-body gap = 2 * min over allowed k of |f(k)|.

    SCOPE (validated against ED): this momentum-sum closed form is EXACT (to
    machine precision) for d <= 2, and at every pure stabilizer point e_alpha for
    all d. For d >= 3 under PBC the exact finite-size in-sector gap acquires
    parity / longer-range-hopping corrections NOT captured here (the even sector
    is an antiperiodic fermion chain whose finite-L spectrum is not the naive
    plane-wave lattice). For d >= 3, treat this as the thermodynamic reference;
    use `gap_thermo` for the bulk gap that closes at boundaries (this is what the
    boundary-localization tests in section 4.4 use, scored modulo the O(1/L)
    shift). The in-sector ED gap is also 2x this value (a pair of quasiparticles
    must be excited to preserve parity).
    """
    return 2.0 * float(np.min(single_particle_energies(t, L, bc, sector)))


def gap_thermo(t, n_grid=20001):
    """Thermodynamic gap 2 * min over CONTINUOUS k of |f(k)| (reference value).

    Equivalent to 2 * min_{|z|=1} |g(z)|. A fine grid then sits well below the
    finite-size shift; exact zeros are captured by `is_gapless`.
    """
    ks = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    return 2.0 * float(np.min(np.abs(f_symbol(t, ks))))


def ground_energy_bdg(t, L, bc='pbc', sector=+1):
    """Free-fermion ground-state energy in the chosen sector.

    E0 = - sum over allowed k of |f(k)|. The normalization is fixed by the pure
    stabilizer points: at t = e_alpha every |f(k)| = |t_alpha| so E0 = -L*|t_alpha|,
    matching the ED ground energy of -t_alpha sum_i (commuting stabilizers).

    SCOPE (same caveat as `gap`): EXACT vs ED for d <= 2 and at all-d pure points;
    for d >= 3 under PBC the exact finite-size even-sector energy carries
    parity / boundary-term corrections this closed form omits, so there it is a
    thermodynamic reference, not an exact finite-L value. Validation section 4.1
    checks the exact cases against ED; the d >= 3 free-fermion energy bookkeeping
    is intentionally out of scope for now.
    """
    return -float(np.sum(single_particle_energies(t, L, bc, sector)))


# ----------------------------------------------------------------------------
# Closed-form boundary taxonomy
# ----------------------------------------------------------------------------
def boundary_value_z_plus1(t):
    """g(+1) = sum_alpha t_alpha. Zero on the z=+1 (k=0) boundary hyperplane."""
    return float(np.sum(np.asarray(t, dtype=float)))


def boundary_value_z_minus1(t):
    """g(-1) = sum_alpha (-1)^alpha t_alpha. Zero on the z=-1 (k=pi) boundary."""
    t = np.asarray(t, dtype=float)
    signs = (-1.0) ** np.arange(t.shape[0])
    return float(np.sum(signs * t))


# Static description of the three closed-form boundary types.
BOUNDARY_TYPES = {
    'z=+1': dict(hyperplane='sum_alpha t_alpha = 0', kstar=0.0,
                 c=0.5, delta_omega=(+1, -1),
                 note='real root crosses z=+1; gap closes at k=0'),
    'z=-1': dict(hyperplane='sum_alpha (-1)^alpha t_alpha = 0', kstar=np.pi,
                 c=0.5, delta_omega=(+1, -1),
                 note='real root crosses z=-1; gap closes at k=pi'),
    'complex': dict(hyperplane='conjugate pair |z|=1 at e^{+-i k*}', kstar=None,
                    c=1.0, delta_omega=(+2, -2),
                    note='curved sheet; closing momentum k* returned per crossing'),
}


def _closing_kstar(t):
    """Momentum k* of the root nearest the unit circle (predicted closing k).

    Returns the argument in (0, pi] of the root with |z| closest to 1. For a
    real root near +1 this is ~0; near -1 it is ~pi; for a complex pair it is
    the genuine k* of the crossing.
    """
    roots = g_roots(t)
    if roots.size == 0:
        return None
    idx = int(np.argmin(np.abs(np.abs(roots) - 1.0)))
    return float(abs(np.angle(roots[idx])))


def boundary_crossings(t_path, s_values=None, tol=1e-6, ang_tol=1e-3):
    """Analytic boundary crossings along a path in t-space.

    t_path   : (M, d) array of t-points sampled along a path.
    s_values : (M,) optional path parameter (defaults to arange(M)); crossings
               are reported at the midpoint of the bracketing segment.
    ang_tol  : how close k* must be to 0 or pi to call a crossing "real".

    For each segment where the winding number changes, the crossing is
    classified by the ARGUMENT k* of the root that sits on the unit circle at
    the segment midpoint (robust against the winding-change segment being
    off-by-one from a g(+-1) sign-change segment when a root lands exactly on a
    grid point):
      - z=+1   : k* ~ 0     -> kstar=0,   c=0.5, |dw|=1
      - z=-1   : k* ~ pi    -> kstar=pi,  c=0.5, |dw|=1
      - complex: 0 < k* < pi -> kstar=arg(root), c=1, |dw|=2

    Returns a list of dicts:
      {'s', 'seg', 'type', 'kstar', 'c', 'delta_omega', 'omega_before',
       'omega_after'}.
    Scored modulo the known O(1/L) finite-size shift -- these are the exact
    thermodynamic crossings.
    """
    t_path = np.asarray(t_path, dtype=float)
    M = t_path.shape[0]
    if s_values is None:
        s_values = np.arange(M, dtype=float)
    else:
        s_values = np.asarray(s_values, dtype=float)

    omegas = np.array([winding(t, tol) for t in t_path])

    crossings = []
    for i in range(M - 1):
        dw = int(omegas[i + 1] - omegas[i])
        if dw == 0:
            continue
        s_mid = 0.5 * (s_values[i] + s_values[i + 1])
        t_mid = 0.5 * (t_path[i] + t_path[i + 1])
        # k* of the root nearest the unit circle at the midpoint classifies the
        # crossing type directly and gives the predicted closing momentum.
        kstar = _closing_kstar(t_mid)
        if kstar is not None and kstar < ang_tol:
            ctype, kstar, c = 'z=+1', 0.0, 0.5
        elif kstar is not None and abs(kstar - np.pi) < ang_tol:
            ctype, kstar, c = 'z=-1', float(np.pi), 0.5
        else:
            ctype, c = 'complex', 1.0
        crossings.append({
            's': float(s_mid),
            'seg': i,
            'type': ctype,
            'kstar': kstar,
            'c': c,
            'delta_omega': dw,
            'omega_before': int(omegas[i]),
            'omega_after': int(omegas[i + 1]),
        })
    return crossings
