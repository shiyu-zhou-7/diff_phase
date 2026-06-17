"""
Frozen, t-independent change-of-basis onto a fixed Z2-parity sector.

The global parity is P = prod_i X_i. In the computational (Z) basis X flips a
bit, so P is the global bit-flip permutation:  P|s> = |flip(s)>, where flip(s)
inverts all L bits. flip is an involution with NO fixed points (flipping all L
bits always changes the string for L >= 1), so the 2**L basis states fall into
exactly 2**(L-1) pairs {s, flip(s)}.

The +-1 parity eigenvectors are the symmetric / antisymmetric combinations:
    sector = +1 :  (|s> + |flip(s)>)/sqrt(2)
    sector = -1 :  (|s> - |flip(s)>)/sqrt(2)

We build the (2**L, 2**(L-1)) isometry V whose columns are these combinations.
P depends ONLY on L (never on t), so V is constructed once and treated as a
constant -- it must never land on the autodiff tape (that would inject a
spurious connection term into state gradients).

Returns a small `ParitySector` bundle carrying:
  - `V`            : scipy.sparse isometry (full <- sector), for projecting
                     operators:  O_sector = V.T @ O_full @ V.
  - lift index arrays (`lift_rows`, `lift_cols`, `lift_vals`): a JAX-friendly
    scatter that maps a sector state vector to its full 2**L vector
    (psi_full = scatter(lift_vals * psi_sector[lift_cols] -> lift_rows)),
    used by the observable layer (differentiable, no dense V needed).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class ParitySector:
    L: int
    sector: int            # +1 or -1
    dim_full: int          # 2**L
    dim_sector: int        # 2**(L-1)
    reps: np.ndarray       # (2**(L-1),) representative basis index of each pair
    flips: np.ndarray      # (2**(L-1),) flip(reps)
    V: sp.csr_matrix       # (2**L, 2**(L-1)) isometry full <- sector
    # JAX scatter for lifting a sector vector to the full space:
    lift_rows: np.ndarray  # (2 * dim_sector,)
    lift_cols: np.ndarray  # (2 * dim_sector,)
    lift_vals: np.ndarray  # (2 * dim_sector,)


def build_parity_sector(L, sector=+1):
    """Construct the frozen P=`sector` change-of-basis for an L-site chain."""
    if sector not in (+1, -1):
        raise ValueError(f'sector must be +1 or -1, got {sector!r}')
    dim_full = 1 << L
    all_idx = np.arange(dim_full, dtype=np.int64)
    flip_all = (1 << L) - 1
    flipped = all_idx ^ flip_all

    # representative = min(s, flip(s)); keep each pair once
    reps = all_idx[all_idx < flipped]
    flips = reps ^ flip_all
    dim_sector = reps.shape[0]
    assert dim_sector == dim_full // 2, "parity pairing is not 2**(L-1)"

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    cols = np.arange(dim_sector, dtype=np.int64)

    # V columns: (|rep> + sector|flip>)/sqrt2
    rows = np.concatenate([reps, flips])
    col_idx = np.concatenate([cols, cols])
    vals = np.concatenate([
        np.full(dim_sector, inv_sqrt2),
        np.full(dim_sector, sector * inv_sqrt2),
    ])
    V = sp.csr_matrix((vals, (rows, col_idx)), shape=(dim_full, dim_sector))

    return ParitySector(
        L=L, sector=sector, dim_full=dim_full, dim_sector=dim_sector,
        reps=reps, flips=flips, V=V,
        lift_rows=rows, lift_cols=col_idx, lift_vals=vals,
    )
