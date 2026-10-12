"""Replot manuscript data with NumPy and Matplotlib only.
Usage: python plot_all.py [--output /path/to/figures]
"""
from pathlib import Path
import argparse
import runpy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'scripts'))
    panels=['tfim_parameter','z2_parameter','xxz_parameter','tfim_latent','z2_latent','xxz_latent']
    for name in panels+['benchmarks']:
        with plt.rc_context():
            runpy.run_path(str(ROOT/'scripts'/f'{name}.py'),init_globals={'OUTPUT':out})
        print('Plotted',name)
    # A portable PNG overview; individual panels above remain vector PDFs.
    images=[plt.imread(out/f'{name}.png') for name in panels]
    h,w=images[0].shape[:2]
    fig=plt.figure(figsize=(19.5,9.6),dpi=160)
    for i,img in enumerate(images):
        ax=fig.add_axes([(i%3)/3, (1-i//3)*0.46875,1/3,0.46875])
        ax.imshow(img);ax.set_axis_off()
    for i,title in enumerate(['TFIM',r'$Z_2$','XXZ']):
        fig.text((i+0.5)/3,0.96,title,ha='center',va='center',fontsize=28)
    fig.savefig(out/'demonstrations.png');plt.close(fig)
    print('Figures written to',out)

if __name__=='__main__': main()
