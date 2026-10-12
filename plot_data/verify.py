"""Validate portable arrays, PCA projections, checksums, and benchmark summaries."""
from pathlib import Path
import collections
import hashlib
import json
import statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,info in manifest['data_files'].items():
        p=ROOT/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'],name
    for p in (ROOT/'data').glob('*.npz'):
        with np.load(p,allow_pickle=False) as d:
            meta=json.loads(p.with_suffix('.json').read_text())
            assert set(d.files)==set(meta['arrays']),p
            for key in d.files:
                a=d[key]
                assert not a.dtype.hasobject,(p,key)
                assert list(a.shape)==meta['arrays'][key]['shape'],(p,key)
                if np.issubdtype(a.dtype,np.number): assert np.isfinite(a).all(),(p,key)
            if 'train_pca' in d:
                z=d['z_sub'] if 'z_sub' in d else d['Z_all'] if 'Z_all' in d else d['Z_train']
                np.testing.assert_allclose((z-d['z_mean'])@d['pca_components'][:2].T,d['train_pca'],atol=1e-12)
                np.testing.assert_allclose((d['traj_latents']-d['z_mean'])@d['pca_components'][:2].T,d['traj_pca'],atol=1e-12)
                assert len(d['train_phase'])==len(d['train_pca'])
            for key in d.files:
                if key.endswith('_normalized_step'):
                    prefix=key.removesuffix('_normalized_step')
                    n=len(d[prefix+'_values'])
                    assert len(d[prefix+'_event'])==n
                    np.testing.assert_allclose(d[key],np.linspace(0,1,n))
    b=ROOT/'data/benchmarks'
    groups=collections.defaultdict(list)
    for p in (b/'final_res').glob('*.json'):
        j=json.loads(p.read_text());groups[j['d'],j['budget']].append(j['cov'])
    u=json.loads((b/'unif_all.json').read_text())
    assert len(groups)==40 and all(len(v)==5 for v in groups.values())
    separated=sum(min(v)>max(u[str(budget)][str(d)]) for (d,budget),v in groups.items())
    assert separated==36
    aa=[];uu=[]
    for seed in range(200,205):
        a=set();u0=set()
        for chunk in range(10):
            j=json.loads((b/f'chern_res_M768/relab_s{seed}_c{chunk}.json').read_text())
            assert j['agent']['n']==j['uniform']['n']==30000
            a.update(j['agent']['set_conv']);u0.update(j['uniform']['set_conv'])
        aa.append(a);uu.append(u0)
    assert st.median(map(len,aa))==207 and st.median(map(len,uu))==139
    a=set.union(*aa);u0=set.union(*uu)
    v=set(map(int,json.loads((b/'sector_volumes_M768.json').read_text())))
    assert (len(a),len(u0),len(u0|v),len(a-(u0|v)))==(283,160,168,115)
    assert not (u0|v)-a
    print('PASS: all data hashes, six panel schemas, PCA projections, trajectory indices, and benchmark summaries.')

if __name__=='__main__': main()
