"""Generalized 2D Chern insulator (higher-harmonic QWZ), 6D parameter space.

  d_x = sin kx + a sin 2kx
  d_y = sin ky + b sin 2ky
  d_z = m + t1(cos kx + cos ky) + t2(cos 2kx + cos 2ky) + t3 cos kx cos ky
  theta = (m, a, b, t1, t2, t3)

Chern number by the Fukui-Hatsugai-Suzuki lattice method (gauge invariant,
integer by construction).
"""
import numpy as np

def dvec(theta, KX, KY):
    m, a, b, t1, t2, t3 = theta
    dx = np.sin(KX) + a*np.sin(2*KX)
    dy = np.sin(KY) + b*np.sin(2*KY)
    dz = (m + t1*(np.cos(KX) + np.cos(KY)) + t2*(np.cos(2*KX) + np.cos(2*KY))
          + t3*np.cos(KX)*np.cos(KY))
    return dx, dy, dz


def lower_band(dx, dy, dz):
    """Lower-band eigenvector of d.sigma.  Two equivalent forms are available;
    each degenerates on one hemisphere, so pick the well-conditioned one:
       dz <= 0 :  u = (dz - n,  dx + i dy)
       dz >  0 :  u = (dx - i dy, -(dz + n))
    FHS is gauge invariant, so switching convention between k-points is safe."""
    n = np.sqrt(dx**2 + dy**2 + dz**2)
    nz = np.maximum(n, 1e-300)
    a0, a1 = dz - nz, dx + 1j*dy                 # good for dz <= 0
    b0, b1 = dx - 1j*dy, -(dz + nz)              # good for dz >  0
    use_b = dz > 0
    u0 = np.where(use_b, b0, a0)
    u1 = np.where(use_b, b1, a1)
    nrm = np.sqrt(np.abs(u0)**2 + np.abs(u1)**2)
    bad = nrm < 1e-12                            # |d| = 0: gapless, undefined
    u0 = np.where(bad, 1.0, u0); u1 = np.where(bad, 0.0, u1)
    nrm = np.maximum(nrm, 1e-300)
    return u0/nrm, u1/nrm


def chern(theta, M=64):
    k = 2*np.pi*np.arange(M)/M
    KX, KY = np.meshgrid(k, k, indexing="ij")
    dx, dy, dz = dvec(theta, KX, KY)
    gap = np.sqrt(dx**2 + dy**2 + dz**2).min()
    u0, u1 = lower_band(dx, dy, dz)
    def link(ax):
        v0 = np.roll(u0, -1, axis=ax); v1 = np.roll(u1, -1, axis=ax)
        ov = np.conj(u0)*v0 + np.conj(u1)*v1
        return ov/np.maximum(np.abs(ov), 1e-300)
    Ux, Uy = link(0), link(1)
    F = np.log(Ux * np.roll(Uy, -1, axis=0) /
               np.roll(Ux, -1, axis=1) / Uy).imag
    tot = F.sum()
    if not np.isfinite(tot):
        return None, float(gap)          # gapless / undefined
    return int(np.rint(tot/(2*np.pi))), float(gap)


if __name__ == "__main__":
    print("  校准: 标准 QWZ (a=b=t2=t3=0, t1=1) —— 已知 C = 0/-1/+1/0")
    for m in [-3, -1, 1, 3]:
        c, g = chern((m, 0, 0, 1, 0, 0))
        print(f"    m={m:+3.0f}: C={c:+2d}  gap={g:.3f}")
    print("\n  高次谐波能产生 |C|>=2 吗?")
    for th in [(0, 1.5, 1.5, 1, 0.5, 0), (0.5, 2, 2, 0.5, 1, 0),
               (0, 0, 0, 0, 1, 0), (1, 3, 3, 1, 1.5, 0.5)]:
        c, g = chern(th)
        print(f"    theta={th}: C={c:+2d}  gap={g:.3f}")
