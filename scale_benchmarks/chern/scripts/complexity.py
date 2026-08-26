"""Complexity structure of the Chern inverse problem for Eq.(10).

KEY: at FIXED root positions, d_z is LINEAR in the 18-vector (c_1..c_H, m, e).
So the achievable sign patterns on the p x q grid are exactly the cells of an
arrangement of pq hyperplanes in R^18  ->  at most sum_{k<=18} C(pq,k) cells,
POLYNOMIAL in H, not 2^{pq}.
"""
import numpy as np, sys, itertools
from math import comb
sys.path.insert(0, ".")
sys.path.insert(0, "/private/tmp/claude-501/-Users-vmac-Downloads/4c8d2334-da28-4e82-a2ff-1a13aadace55/scratchpad")
from degree import roots_of_sin_series
from chern_fast import chern_fast
H = 16; D = H + 2                     # (c_1..c_H, m, e)

print("=== 1. sign patterns: naive vs arrangement bound ===")
for p, q in [(8, 8), (16, 16), (32, 32)]:
    log10naive = p*q*np.log10(2)
    arr = sum(comb(p*q, k) for k in range(D+1))
    print(f"  grid {p:2d}x{q:2d} = {p*q:4d} pts:  naive 2^(pq) = 1e{log10naive:.0f}"
          f"   arrangement cells <= 1e{np.log10(float(arr)):.1f}  (saving 1e{log10naive-np.log10(float(arr)):.0f})")

print("\n=== 2. verify d_z is linear in (c,m,e) at fixed grid ===")
rng = np.random.default_rng(11)
a = rng.standard_normal(H); b = rng.standard_normal(H)
kx = roots_of_sin_series(a); ky = roots_of_sin_series(b)
n = np.arange(1, H+1)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
# design matrix: d_z(grid) = A @ (c, m, e)
A = np.concatenate([ (np.cos(n*KX[...,None]) + np.cos(n*KY[...,None])).reshape(-1, H),
                     np.ones((KX.size,1)), (np.cos(KX)*np.cos(KY)).reshape(-1,1)], axis=1)
v1, v2 = rng.standard_normal(D), rng.standard_normal(D)
lhs = A @ (0.3*v1 + 0.7*v2); rhs = 0.3*(A@v1) + 0.7*(A@v2)
print(f"  grid {len(kx)}x{len(ky)} = {A.shape[0]} points, design matrix {A.shape}")
print(f"  linearity residual: {np.abs(lhs-rhs).max():.2e}   rank(A) = {np.linalg.matrix_rank(A)}")

print("\n=== 3. reachable C at FIXED (a,b): sample the 18-dim cell space ===")
sx = np.sign(np.sum(a*n*np.cos(np.outer(kx,n)),axis=1))
sy = np.sign(np.sum(b*n*np.cos(np.outer(ky,n)),axis=1))
S = np.outer(sx, sy).ravel()
seen = {}
for t in range(200000):
    w = rng.standard_normal(D)
    dz = A @ w
    C = int(-np.sum(S[dz > 0]))
    seen.setdefault(C, 0); seen[C] += 1
ks = sorted(seen)
print(f"  samples 2e5 -> {len(seen)} distinct C, range [{ks[0]}, {ks[-1]}]")
print(f"  (this single (a,b) grid, varying only the 18 remaining parameters)")
print(f"  most common: {sorted(seen.items(), key=lambda x:-x[1])[:6]}")
