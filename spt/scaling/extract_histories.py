"""
Strip the L=12 scaling bundles down to the per-step winding trail.

For each non-checkpoint spt_active_*.pkl in <data_dir>, keep only what the
manuscript figures need -- d, seed, t_init, omega/query/event per step -- and
write one compact JSON. Run on the cluster (bundles hold jnp arrays, so the
jax-capable conda env is required to unpickle), then scp the JSON home.

Usage:
    python extract_histories.py <data_dir> <out.json>
"""
import glob
import json
import os
import pickle
import sys


def main(data_dir, out_path):
    recs = []
    for path in sorted(glob.glob(os.path.join(data_dir, 'spt_active_*.pkl'))):
        if path.endswith('_ckpt.pkl'):
            continue
        with open(path, 'rb') as f:
            b = pickle.load(f)
        cfg = b['cfg_snapshot']
        h = b['history']
        recs.append(dict(
            d=cfg['cluster_cfg']['d'],
            seed=cfg['cluster_cfg']['seed'],
            t_init=cfg['t_init'],
            total_queries=b['total_queries'],
            omega_per_step=[int(w) for w in h['omega_per_step']],
            query_per_step=[int(q) for q in h['query_per_step']],
            event_per_step=list(h['event_per_step']),
        ))
    with open(out_path, 'w') as f:
        json.dump(recs, f)
    print(f'{len(recs)} runs -> {out_path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
