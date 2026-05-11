"""
Small plotting helpers for TFIM main_*.py scripts. Each function produces a
single PDF/PNG figure and returns the path written.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — avoids display hangs in batch runs
import matplotlib.pyplot as plt


def plot_loss_curve(losses, out_path, ylabel='loss', logy=True, title=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(losses, 'o', color='blue', markersize=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_h_trajectory(h_list, out_path, title=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    hs = np.asarray(h_list)
    ax.plot(range(len(hs)), hs, 'o-', color='blue', markersize=2, lw=1)
    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$h$')
    if len(hs) > 0:
        h_min, h_max = float(hs.min()), float(hs.max())
        h_range = h_max - h_min
        pad = 0.1 * h_range if h_range > 0 else 0.1
        ax.set_ylim(h_min - pad, h_max + pad)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_h_loss_combined(h_list, loss_list, out_path, title=None):
    """Two-panel: h vs epoch on top, loss vs epoch on bottom."""
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
    axes[0].plot(h_list, 'o', color='blue', markersize=2)
    axes[0].set_ylabel(r'$h$')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss_list, 'o', color='orange', markersize=2)
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].grid(True, alpha=0.3)

    if title is not None:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_latent_scatter(latent_ssb, latent_para, out_path, dims=(0, 1), title=None):
    """
    Scatter the first two latent components of two phase-labeled batches.
    `latent_ssb` and `latent_para` are arrays of shape [N, latent_dim].
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    a, b = dims
    ax.scatter(np.asarray(latent_ssb)[:, a], np.asarray(latent_ssb)[:, b],
               s=10, alpha=0.5, color='tab:blue', label='ssb')
    ax.scatter(np.asarray(latent_para)[:, a], np.asarray(latent_para)[:, b],
               s=10, alpha=0.5, color='tab:orange', label='para')
    ax.set_xlabel(f'latent dim {a}')
    ax.set_ylabel(f'latent dim {b}')
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path
