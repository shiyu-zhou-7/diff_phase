"""Successive halving (Hyperband bracket) over the pipeline's hyperparameters.

The resource is the query budget.  Unlike NN training, an agent run is
naturally *continuable*: a trial that survives a rung resumes its walk from
where it stopped, so promoted trials waste nothing.

  rung r:  n_r = n0 / eta^r  trials, each run to total budget B0 * eta^r
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/diff_phase/spt/scripts")
import jax, jax.numpy as jnp, optax
jax.config.update("jax_enable_x64", True)
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
sys.path.insert(0, "/scratch/yuxzhang/chern")
from chern_fast import chern_fast

H = int(os.environ.get("CH_H", "16"))
DIM = 3*H + 2
M = max(32, 3*H//2 + 8)
M_LAB = max(96, 4*H)
SEED = int(os.environ.get("CH_SEED", "0"))
N0 = int(os.environ.get("HB_N0", "27"))
B0 = int(os.environ.get("HB_B0", "2500"))
ETA = 3
BOOT_N, AE_HID, LATENT, AE_EPOCHS = 30, 32, 16, 1500

_k = 2*jnp.pi*jnp.arange(M)/M
KX, KY = jnp.meshgrid(_k, _k, indexing="ij")
_n = jnp.arange(1, H+1)[:, None, None]
SX, SY = jnp.sin(_n*KX), jnp.sin(_n*KY)
CZ = jnp.cos(_n*KX) + jnp.cos(_n*KY); CC = jnp.cos(KX)*jnp.cos(KY)

def feat(th):
    dx = jnp.tensordot(th[:H], SX, 1); dy = jnp.tensordot(th[H:2*H], SY, 1)
    dz = th[3*H] + jnp.tensordot(th[2*H:3*H], CZ, 1) + th[3*H+1]*CC
    n = jnp.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
    v = jnp.concatenate([(dx/n).ravel(), (dy/n).ravel(), (dz/n).ravel()])
    return v/jnp.linalg.norm(v)
featb = jax.jit(jax.vmap(feat))
FDIM = 3*M*M

MEMK = 50          # fixed-size trail buffer -> one JIT compilation only

def _loss(t, p, cen, mem, tw, ts):
    z = fetch_latent(p, feat(t)[None, :], jax.random.PRNGKey(0))[0]
    v = -jnp.sum((z-cen)**2)
    d2 = jnp.sum((mem - t[None, :])**2, axis=1)
    return v + tw*jnp.sum(jnp.exp(-d2/ts**2))

GRAD = jax.jit(jax.value_and_grad(_loss, argnums=0))

def trail_buffer(traj, horizon, dim):
    """Always MEMK rows: the last `horizon` steps, evenly subsampled/padded."""
    if len(traj) < 2:
        return np.full((MEMK, dim), 1e6)
    h = np.asarray(traj[-horizon:])
    idx = np.linspace(0, len(h)-1, MEMK).astype(int)
    return h[idx]

def label(th):
    c, g = chern_fast(np.asarray(th), H, M=M_LAB)
    return c if (c is not None and g > 1e-6) else None

def sample_cfg(r):
    lg = lambda lo, hi: float(np.exp(r.uniform(np.log(lo), np.log(hi))))
    return dict(lr=lg(0.01, 3.0), trail_w=lg(0.1, 20.0), trail_s=lg(0.02, 2.0),
                trail_n=int(lg(100, 50000)), lazy=int(lg(1, 300)),
                radius=lg(0.02, 1.0), max_steps=int(lg(100, 5000)),
                clip=lg(0.05, 5.0))

class Trial:
    """A continuable agent run."""
    def __init__(self, cfg, seed):
        self.c = cfg; self.q = 0; self.seen = set(); self.traj = []
        self.rng = np.random.default_rng(seed); self.key = jax.random.PRNGKey(seed)
        v = np.zeros(DIM); v[3*H] = 1.0; v /= np.linalg.norm(v)
        self.t = jnp.asarray(v); self.p = None; self.cen = None; self.si = 0
        self.g = None
        self._boot(np.asarray(self.t))

    def _boot(self, c):
        P = c[None, :] + self.c["radius"]*self.rng.standard_normal((BOOT_N, DIM))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for s in P:
            lab = label(s); self.q += 1
            if lab is not None: self.seen.add(lab)
        X = featb(jnp.asarray(P))
        self.key, kk = jax.random.split(self.key)
        p = init_params([FDIM, AE_HID, LATENT, AE_HID, FDIM], kk, scale=1e-2)
        p = train_autoencoder(p, X, epochs=AE_EPOCHS, lr=1e-3, center_coeff=1e-3,
                              seed=83948, loss='fidelity', log_every=10**9)
        if isinstance(p, tuple): p = p[0]
        self.p = p
        self.cen = jnp.mean(fetch_latent(p, X, jax.random.PRNGKey(0)), axis=0)

    def run_to(self, budget):
        c = self.c
        while self.q < budget:
            sched = optax.cosine_decay_schedule(c["lr"], c["max_steps"], alpha=0.02)
            opt = optax.chain(optax.clip_by_global_norm(c["clip"]), optax.adam(sched))
            st = opt.init(self.t); recent = []
            for _ in range(c["max_steps"]):
                if self.si % c["lazy"] == 0:
                    mem = jnp.asarray(trail_buffer(self.traj, c["trail_n"], DIM))
                    _, self.g = GRAD(self.t, self.p, self.cen, mem,
                                     c["trail_w"], c["trail_s"])
                    self.q += 1
                    lab = label(np.asarray(self.t))
                    if lab is not None: self.seen.add(lab)
                    self.traj.append(np.asarray(self.t))
                tp = self.t
                upd, st = opt.update(self.g, st, self.t)
                self.t = optax.apply_updates(self.t, upd)
                self.t = self.t/jnp.linalg.norm(self.t); self.si += 1
                recent.append(float(jnp.max(jnp.abs(self.t-tp))))
                if len(recent) > 20: recent.pop(0)
                if len(recent) == 20 and max(recent) < 1e-3: break
                if self.q >= budget: break
            if self.q >= budget: break
            if len(recent) == 20 and max(recent) < 1e-3:
                self._boot(np.asarray(self.t))
        return len(self.seen)


if __name__ == "__main__":
    t00 = time.time()
    rng = np.random.default_rng(1000 + SEED)
    trials = [Trial(sample_cfg(rng), SEED*100 + i) for i in range(N0)]
    alive = list(range(N0))
    rung = 0
    hist = []
    while len(alive) > 1:
        budget = B0 * (ETA ** rung)
        scores = []
        for i in alive:
            sc = trials[i].run_to(budget)
            scores.append((sc, i))
        scores.sort(reverse=True)
        keep = max(1, len(alive)//ETA)
        best = scores[0]
        hist.append(dict(rung=rung, budget=budget, n=len(alive),
                         best=best[0], best_cfg=trials[best[1]].c,
                         all=[s for s, _ in scores]))
        print(f"  rung {rung}: n={len(alive):3d} budget={budget:7,d} "
              f"best={best[0]:4d}  [{time.time()-t00:.0f}s]", flush=True)
        alive = [i for _, i in scores[:keep]]
        rung += 1
    final_budget = B0 * (ETA ** rung)
    win = alive[0]
    sc = trials[win].run_to(final_budget)
    # matched-budget random baseline
    rb = np.random.default_rng(SEED); U = set()
    for _ in range(final_budget):
        c, g = chern_fast(rb.standard_normal(DIM), H, M=M_LAB)
        if c is not None and g > 1e-6: U.add(c)
    print(f"\n  WINNER at budget {final_budget:,}: agent {sc}  vs random {len(U)}"
          f"   diff {sc-len(U):+d}")
    print(f"  config: { {k: (round(v,4) if isinstance(v,float) else v) for k,v in trials[win].c.items()} }")
    json.dump(dict(H=H, dim=DIM, seed=SEED, final_budget=final_budget,
                   agent=sc, uniform=len(U), cfg=trials[win].c, hist=hist),
              open(f"hb_H{H}_s{SEED}.json", "w"), indent=1, default=str)
