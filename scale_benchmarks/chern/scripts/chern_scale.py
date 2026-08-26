"""Can the Chern family be made rich?  Add harmonics up to order H:
   d_x = sum_{n=1..H} a_n sin(n kx),   d_y = sum_n b_n sin(n ky)
   d_z = m + sum_n [ c_n (cos n kx + cos n ky) ] + e (cos kx cos ky)
parameter count = 2H + H + 2 = 3H + 2
"""
import numpy as np

def chern_H(theta, H, M=96):
    k = 2*np.pi*np.arange(M)/M
    KX, KY = np.meshgrid(k, k, indexing="ij")
    a = theta[:H]; b = theta[H:2*H]; c = theta[2*H:3*H]; m, e = theta[3*H], theta[3*H+1]
    dx = sum(a[n]*np.sin((n+1)*KX) for n in range(H))
    dy = sum(b[n]*np.sin((n+1)*KY) for n in range(H))
    dz = m + sum(c[n]*(np.cos((n+1)*KX)+np.cos((n+1)*KY)) for n in range(H)) \
         + e*np.cos(KX)*np.cos(KY)
    n_ = np.sqrt(dx**2+dy**2+dz**2); gap = n_.min()
    if gap < 1e-6: return None, gap
    nz = np.maximum(n_, 1e-300)
    a0, a1 = dz-nz, dx+1j*dy
    b0, b1 = dx-1j*dy, -(dz+nz)
    ub = dz > 0
    u0 = np.where(ub, b0, a0); u1 = np.where(ub, b1, a1)
    nr = np.maximum(np.sqrt(np.abs(u0)**2+np.abs(u1)**2), 1e-300)
    u0, u1 = u0/nr, u1/nr
    def link(ax):
        v0 = np.roll(u0,-1,axis=ax); v1 = np.roll(u1,-1,axis=ax)
        ov = np.conj(u0)*v0 + np.conj(u1)*v1
        return ov/np.maximum(np.abs(ov),1e-300)
    Ux, Uy = link(0), link(1)
    F = np.log(Ux*np.roll(Uy,-1,axis=0)/np.roll(Ux,-1,axis=1)/Uy).imag
    t = F.sum()
    return (int(np.rint(t/(2*np.pi))) if np.isfinite(t) else None), float(gap)

if __name__ == "__main__":
    print("   H | 参数维数 | 均匀采样 3000 点找到的 C 值 | 个数 | 最稀相占比")
    for H in [1, 2, 3, 4, 6, 8]:
        dim = 3*H + 2
        rng = np.random.default_rng(0); C = []
        for _ in range(3000):
            c, g = chern_H(rng.standard_normal(dim), H)
            if c is not None and g > 1e-3: C.append(c)
        u, n = np.unique(C, return_counts=True)
        pm = n.min()/len(C)
        rng_str = f"[{u.min()},{u.max()}]"
        print(f"  {H:2d} | {dim:8d} | {rng_str:12s} | {len(u):4d} | {pm:.4f}", flush=True)
