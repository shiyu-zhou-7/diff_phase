"""Cached-basis Chern evaluation: the sin/cos basis on the BZ grid depends only
on (H, M), so build it once instead of recomputing inside every call."""
import numpy as np
_B = {}

def basis(H, M):
    if (H, M) not in _B:
        k = 2*np.pi*np.arange(M)/M
        KX, KY = np.meshgrid(k, k, indexing="ij")
        n = np.arange(1, H+1)[:, None, None]
        _B[(H, M)] = (np.sin(n*KX), np.sin(n*KY),
                      np.cos(n*KX) + np.cos(n*KY), np.cos(KX)*np.cos(KY))
    return _B[(H, M)]

def chern_fast(theta, H, M=None):
    M = M or max(96, 4*H)
    SX, SY, CZ, CC = basis(H, M)
    a, b, c = theta[:H], theta[H:2*H], theta[2*H:3*H]
    m, e = theta[3*H], theta[3*H+1]
    dx = np.tensordot(a, SX, 1); dy = np.tensordot(b, SY, 1)
    dz = m + np.tensordot(c, CZ, 1) + e*CC
    n_ = np.sqrt(dx*dx + dy*dy + dz*dz); gap = n_.min()
    if gap < 1e-9: return None, float(gap)
    ub = dz > 0
    u0 = np.where(ub, dx - 1j*dy, dz - n_)
    u1 = np.where(ub, -(dz + n_), dx + 1j*dy)
    nr = np.sqrt(np.abs(u0)**2 + np.abs(u1)**2); nr[nr < 1e-300] = 1.0
    u0 /= nr; u1 /= nr
    def link(ax):
        ov = np.conj(u0)*np.roll(u0, -1, ax) + np.conj(u1)*np.roll(u1, -1, ax)
        aa = np.abs(ov); aa[aa < 1e-300] = 1.0
        return ov/aa
    Ux, Uy = link(0), link(1)
    F = np.angle(Ux*np.roll(Uy, -1, 0)*np.conj(np.roll(Ux, -1, 1))*np.conj(Uy))
    t = F.sum()
    return (int(np.rint(t/(2*np.pi))) if np.isfinite(t) else None), float(gap)

if __name__ == "__main__":
    import time
    from chern_scale import chern_H
    rng = np.random.default_rng(0)
    print("  正确性: 新旧实现对比")
    for H in [6, 16]:
        ok = 0
        for _ in range(40):
            th = rng.standard_normal(3*H+2)
            c1, _ = chern_H(th, H, M=4*H if 4*H > 96 else 96)
            c2, _ = chern_fast(th, H, M=4*H if 4*H > 96 else 96)
            ok += (c1 == c2)
        print(f"    H={H}: {ok}/40 一致")
    print("\n  加速后的成本")
    print("     H |  dim |    M | 单次标注 | 2万预算")
    for H in [6, 16, 32, 66, 100]:
        M = max(96, 4*H); dim = 3*H+2
        th = rng.standard_normal(dim); chern_fast(th, H, M)
        t0 = time.time(); [chern_fast(th, H, M) for _ in range(5)]; dt = (time.time()-t0)/5
        print(f"    {H:3d} | {dim:4d} | {M:4d} | {1000*dt:7.1f}ms | {20000*dt/60:6.1f} min")
