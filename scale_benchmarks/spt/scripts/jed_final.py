"""Production runs for the cluster-chain benchmark (Sec. IV B of the paper).

Long-range Kitaev chain in class BDI at L = 500, d and budget from the
environment (SW_D, SW_BUDGET), e_0 start, one solve per SW_LAZY parameter
steps.  Ground states are closed form: the NS-sector Majorana matrix is a
twisted circulant, so its polar factor is g(r) = (1/L) sum_k [f(k)/|f(k)|]
e^{-ikr} and no SVD is needed.  Phase labels come from exact root counting
(winding_label_phys_g) and never enter the objective.
Writes final_res/f_d<d>_b<budget>_s<seed>.json."""
import sys, time, json, itertools
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/diff_phase/spt/scripts")
import jax, jax.numpy as jnp, optax
jax.config.update("jax_enable_x64", True)
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
sys.path.insert(0, "/scratch/yuxzhang/hpscan")
from cluster_ed_general import winding_label_phys_g
from fair_k_analysis import wind_batch

import os
L, NRUN = 500, 1
Q = int(os.environ['SW_BUDGET'])
D = int(os.environ['SW_D'])
LAZY = int(os.environ['SW_LAZY'])
SEED = int(os.environ['SW_SEED'])
START = os.environ.get('SW_START','e0')
BOOT_N, RADIUS, AE_HID, LATENT, AE_EPOCHS = 30, 0.1, 32, 16, 800
DEF = dict(lr=5e-2, clip=0.5, max_steps=500, trail_w=0.5, trail_s=0.15,
           trail_n=400, lazy=1, radius=0.1)
n_ = jnp.arange(L); k_ = (2*n_+1)*jnp.pi/L; a_ = jnp.arange(D); r_ = jnp.arange(L)
EF = jnp.exp(1j*k_[:, None]*a_[None, :]); ER = jnp.exp(-1j*k_[:, None]*r_[None, :])
def feat(t):
    c = jnp.concatenate([t[:1], -t[1:]])
    f = (c[None, :]*EF).sum(1)
    return jnp.real(((f/jnp.abs(f))[:, None]*ER).sum(0)/L)
featb = jax.jit(jax.vmap(feat))

def make_loss(tw, ts):
    def loss(t, p, cen, mem):
        z = fetch_latent(p, feat(t)[None, :], jax.random.PRNGKey(0))[0]
        v = -jnp.sum((z-cen)**2)
        if mem is not None:
            v = v + tw*jnp.sum(jnp.exp(-jnp.sum((mem-t[None, :])**2, axis=1)/ts**2))
        return v
    return jax.jit(jax.value_and_grad(loss))
_G = {}
def gradf(tw, ts):
    if (tw, ts) not in _G: _G[(tw, ts)] = make_loss(tw, ts)
    return _G[(tw, ts)]

def run(cfg, seed, start="rand"):
    c = {**DEF, **cfg}
    gf = gradf(c["trail_w"], c["trail_s"])
    rng = np.random.default_rng(seed); key = jax.random.PRNGKey(seed)
    if start == "e0": t = jnp.zeros(D).at[0].set(1.0)
    else:
        v = rng.standard_normal(D); v /= np.linalg.norm(v); t = jnp.asarray(v)
    seen, q = set(), 0
    def boot(cc, kk):
        nonlocal q
        P = np.asarray(cc)[None, :] + c["radius"]*rng.standard_normal((BOOT_N, D))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for s in P: seen.add(winding_label_phys_g(s)[0]); q += 1
        X = featb(jnp.asarray(P))
        p = init_params([L, AE_HID, LATENT, AE_HID, L], kk, scale=1e-2)
        p = train_autoencoder(p, X, epochs=AE_EPOCHS, lr=1e-3, center_coeff=1e-3,
                              seed=83948, loss='fidelity', log_every=10**9)
        if isinstance(p, tuple): p = p[0]
        return p, jnp.mean(fetch_latent(p, X, jax.random.PRNGKey(0)), axis=0)
    key, bk = jax.random.split(key)
    p, cen = boot(np.asarray(t), bk)
    traj = []; g = None; si = 0
    while q < Q:
        sched = optax.cosine_decay_schedule(c["lr"], c["max_steps"], alpha=0.02)
        opt = optax.chain(optax.clip_by_global_norm(c["clip"]), optax.adam(sched))
        st = opt.init(t); recent = []
        for _ in range(c["max_steps"]):
            if si % c["lazy"] == 0:
                mem = (jnp.asarray(np.asarray(traj[-c["trail_n"]::8]))
                       if len(traj) > 5 else None)
                _, g = gf(t, p, cen, mem)
                q += 1; seen.add(winding_label_phys_g(np.asarray(t))[0])
                traj.append(np.asarray(t))
            tp = t
            upd, st = opt.update(g, st, t)
            t = optax.apply_updates(t, upd); t = t/jnp.linalg.norm(t); si += 1
            recent.append(float(jnp.max(jnp.abs(t-tp))))
            if len(recent) > 20: recent.pop(0)
            if len(recent) == 20 and max(recent) < 1e-3: break
            if q >= Q: break
        if q >= Q: break
        if len(recent) == 20 and max(recent) < 1e-3:
            key, rk = jax.random.split(key); p, cen = boot(np.asarray(t), rk)
    return len(seen)

CONFIGS = [("default", {})]
for v in [0.3, 0.5, 1.0]:        CONFIGS.append((f"trail_s={v}", dict(trail_s=v)))
for v in [2.0, 8.0]:             CONFIGS.append((f"trail_w={v}", dict(trail_w=v)))
for v in [2000, 20000]:          CONFIGS.append((f"trail_n={v}", dict(trail_n=v)))
for v in [10, 30]:               CONFIGS.append((f"lazy={v}", dict(lazy=v)))
for v in [2000, 100000]:         CONFIGS.append((f"max_steps={v}", dict(max_steps=v)))
for v in [0.02, 0.3]:            CONFIGS.append((f"radius={v}", dict(radius=v)))
CONFIGS.append(("s=0.5+lazy10",  dict(trail_s=0.5, lazy=10)))
CONFIGS.append(("s=0.5+n20000",  dict(trail_s=0.5, trail_n=20000)))
CONFIGS.append(("s=0.5+lazy10+n20000", dict(trail_s=0.5, lazy=10, trail_n=20000)))
CONFIGS.append(("no_stall(one block)", dict(max_steps=100000, trail_n=20000)))

if __name__ == "__main__":
    t0 = time.time()
    c = run(dict(lazy=LAZY), SEED, START)
    print(f"  d={D} L={L} lazy={LAZY} b={Q} start={START} seed={SEED}"
          f" -> {c}/{D}   [{time.time()-t0:.0f}s]",
          flush=True)
    json.dump(dict(d=D, L=L, lazy=LAZY, budget=Q, start=START, seed=SEED, cov=c),
              open(f"f_d{D}_b{Q}_s{SEED}.json", "w"))
