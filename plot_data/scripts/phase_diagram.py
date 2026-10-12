import numpy as np
from pathlib import Path
D=np.load(Path(__file__).resolve().parents[1]/"data/xxz_parameter.npz",allow_pickle=False)
def hc_of_delta(delta):
    return np.interp(delta,D["boundary_Delta"],D["boundary_h"])

def draw_phase_diagram(
    ax,
    h_range=(-2.1, 2.1),
    delta_range=(-2.1, 2.1),
    show_regions=True,
    region_alpha=0.10,
    label_regions=True,
):
    """
    Draw the XXZ-in-transverse-field phase diagram onto an existing Axes.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes. x-axis is h, y-axis is Delta.
    h_range, delta_range : (float, float)
        Visible window. Used to clip the boundary curves and to size the
        shaded region polygons.
    show_regions : bool
        If True, lightly shade the four phases.
    region_alpha : float
        Alpha for the shaded regions.
    label_regions : bool
        If True, place small labels (1)-(4) in the phases.
    """
    h_min, h_max = h_range
    d_min, d_max = delta_range

    # Sample Delta densely for smooth curves.
    delta_grid = np.linspace(d_min - 0.5, d_max + 0.5, 600)
    hc_pos = hc_of_delta(delta_grid)
    hc_neg = -hc_pos

    # ---- Shaded phase regions ----------------------------------------------
    if show_regions:
        # Region 4 (paramagnet): |h| > h_c.
        ax.fill_betweenx(
            delta_grid, hc_pos, h_max + 1.0,
            color="#999999", alpha=region_alpha, linewidth=0, zorder=0,
        )
        ax.fill_betweenx(
            delta_grid, h_min - 1.0, hc_neg,
            color="#999999", alpha=region_alpha, linewidth=0, zorder=0,
        )

        # Region 1 (Neel-Z): Delta > 1, |h| < h_c
        mask1 = delta_grid > 1.0
        ax.fill_betweenx(
            delta_grid[mask1], hc_neg[mask1], hc_pos[mask1],
            color="#4477AA", alpha=region_alpha, linewidth=0, zorder=0,
        )

        # Region 2 (FM-Z): Delta < -1, |h| < h_c
        mask2 = delta_grid < -1.0
        ax.fill_betweenx(
            delta_grid[mask2], hc_neg[mask2], hc_pos[mask2],
            color="#EE6677", alpha=region_alpha, linewidth=0, zorder=0,
        )

        # Region 3 (Neel-Y): -1 < Delta < 1, |h| < h_c
        mask3 = (delta_grid > -1.0) & (delta_grid < 1.0)
        ax.fill_betweenx(
            delta_grid[mask3], hc_neg[mask3], hc_pos[mask3],
            color="#228833", alpha=region_alpha, linewidth=0, zorder=0,
        )

    # ---- Critical line h_c(Delta) (thick) ----------------------------------
    ax.plot(hc_pos, delta_grid, color="black", linewidth=1.8, zorder=2)
    ax.plot(hc_neg, delta_grid, color="black", linewidth=1.8, zorder=2)

    # ---- Critical line at Delta = +1, |h| < 2 ------------------------------
    ax.plot([-2.0, 2.0], [1.0, 1.0], color="black", linewidth=1.8, zorder=2)

    # ---- Critical line at Delta = -1, |h| < h_c(-1) ------------------------
    hc_at_m1 = float(D["boundary_h_at_minus1"])
    ax.plot([-hc_at_m1, hc_at_m1], [-1.0, -1.0],
            color="black", linewidth=1.8, zorder=2)

    # ---- Gapless segment on h = 0 with |Delta| < 1 -------------------------
    ax.plot([0.0, 0.0], [-1.0, 1.0],
            color="crimson", linewidth=1.8, zorder=2)

    # ---- Region labels ------------------------------------------------------
    if label_regions and show_regions:
        label_kw = dict(fontsize=16, color="black", alpha=0.75,
                        ha="center", va="center", zorder=3)
        ax.text(0.0,  1.55, "(1)", **label_kw)
        ax.text(0.0, -1.75, "(2)", **label_kw)
        ax.text(+0.7, 0.0,  "(3)", **label_kw)
        ax.text(-0.7, 0.0,  "(3)", **label_kw)
        ax.text(h_max - 0.3, -0.55, "(4)", **label_kw)
        ax.text(h_min + 0.3, -0.55, "(4)", **label_kw)
        ax.text(-0.17, 0.2, "(5)", **label_kw)


# ---------------------------------------------------------------------------
