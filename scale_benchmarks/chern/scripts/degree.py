"""Is the Chern number of Eq.(10) a COMBINATORIAL sum over a zero grid?

d_x depends only on k_x, d_y only on k_y.  So the joint zero set {d_x=d_y=0}
is the PRODUCT of the two root sets, and the Jacobian is diagonal.  Degree =
signed count of preimages of the north pole:
    deg = sum_{(i,j): d_z>0} sign(d_x'(kx_i)) * sign(d_y'(ky_j))
and C = -deg.  Verify against FHS.
"""
import numpy as np, sys
sys.path.insert(0, ".")
from chern_fast import chern_fast
H = 16

def roots_of_sin_series(coef):
    """zeros of sum_n coef[n-1] sin(n k) on [0,2pi)"""
    # sum a_n sin(nk) = Im( sum a_n z^n ), z=e^{ik}.  Equivalent polynomial:
    # (1/2i)[ P(z) - P(1/z) ] = 0  ->  z^H ( P(z) - P(1/z) ) = 0, degree 2H
    H_ = len(coef)
    p = np.zeros(2*H_+1, dtype=complex)     # coefficients of z^0..z^{2H}
    for n in range(1, H_+1):
        p[H_+n] += coef[n-1]/(2j)
        p[H_-n] -= coef[n-1]/(2j)
    r = np.roots(p[::-1])
    r = r[np.abs(np.abs(r)-1) < 1e-8]       # keep unit-circle roots
    k = np.mod(np.angle(r), 2*np.pi)
    k = np.sort(k)
    ded = [k[0]] if len(k) else []
    for x in k[1:]:
        if x - ded[-1] > 1e-7: ded.append(x)
    return np.array(ded)

def combinatorial_C(th):
    a, b, c = th[:H], th[H:2*H], th[2*H:3*H]
    m, e = th[3*H], th[3*H+1]
    kx = roots_of_sin_series(a); ky = roots_of_sin_series(b)
    n = np.arange(1, H+1)
    dxp = lambda k: np.sum(a*n*np.cos(np.outer(k, n)), axis=1)   # d_x'(k)
    dyp = lambda k: np.sum(b*n*np.cos(np.outer(k, n)), axis=1)
    sx, sy = np.sign(dxp(kx)), np.sign(dyp(ky))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    dz = m + np.sum(c*np.cos(n*KX[..., None]), axis=-1) \
           + np.sum(c*np.cos(n*KY[..., None]), axis=-1) + e*np.cos(KX)*np.cos(KY)
    S = np.outer(sx, sy)
    deg = np.sum(S[dz > 0])
    return int(-deg), len(kx), len(ky)

rng = np.random.default_rng(3)
print("  FHS(768)   combinatorial   #kx  #ky   grid")
agree = 0
for t in range(12):
    th = rng.standard_normal(3*H+2)
    c_fhs, g = chern_fast(th, H, 768)
    c_comb, nx, ny = combinatorial_C(th)
    ok = (c_fhs == c_comb); agree += ok
    print(f"   {c_fhs:5d}      {c_comb:6d}        {nx:3d}  {ny:3d}  {nx*ny:5d}   {'OK' if ok else 'MISMATCH'}")
print(f"\n  agreement: {agree}/12")
