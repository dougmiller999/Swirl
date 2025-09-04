
import numpy as np
import matplotlib.pyplot as plt

def _centers_from_edges(edges):
    return 0.5*(edges[:-1] + edges[1:])

def plot_pressure_rz(p, r_edges, z_edges, title=None, unit='Pa', scale=1.0):
    Nr, Nz = p.shape
    assert len(r_edges) == Nr+1 and len(z_edges) == Nz+1, "Edges must be one longer than p-dimensions."
    plt.figure(figsize=(7,5))
    pm = plt.pcolormesh(r_edges, z_edges, (p.T)/scale, shading='auto')
    cbar = plt.colorbar(pm); cbar.set_label(f'Pressure [{unit}]')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    if title: plt.title(title)
    plt.tight_layout(); plt.show()

def plot_pressure_rz_with_overlays(p, r_edges, z_edges, title=None, unit='Pa', scale=1.0,
                                   z_free=None, r_core=None, z_free_is_edges=False, r_core_is_edges=False,
                                   show_contours=False, n_contours=12, savepath=None):
    """
    Color plot of pressure with optional overlays:
      - Free-surface curve: z_free(r)
      - Air-core radius: r_core (constant or function of z)

    Parameters
    ----------
    p : (Nr, Nz) array
        Cell-centered pressure.
    r_edges, z_edges : 1D arrays of length Nr+1, Nz+1
        Grid edges.
    z_free : None | float | 1D array | callable
        If float: horizontal free surface at that z.
        If array: z(r) sampled either at r-centers (len Nr) or r-edges (len Nr+1 if z_free_is_edges=True).
        If callable: function z_free(r) evaluated on r-centers.
    r_core : None | float | 1D array | callable
        If float: constant radius; shaded vertical band [0, r_core].
        If array: r(z) sampled either at z-centers (len Nz) or z-edges (len Nz+1 if r_core_is_edges=True).
        If callable: function r_core(z) evaluated on z-centers.
    z_free_is_edges, r_core_is_edges : bool
        Interpret provided arrays as defined on edges rather than centers.
    show_contours : bool
        Overlay pressure contours.
    n_contours : int
        Number of contour levels.
    savepath : str | None
        If given, save figure to this path.
    """
    Nr, Nz = p.shape
    assert len(r_edges) == Nr+1 and len(z_edges) == Nz+1, "Edges must be one longer than p-dimensions."

    r_c = _centers_from_edges(r_edges)
    z_c = _centers_from_edges(z_edges)

    fig = plt.figure(figsize=(7,5))
    pm = plt.pcolormesh(r_edges, z_edges, (p.T)/scale, shading='auto')
    cbar = plt.colorbar(pm); cbar.set_label(f'Pressure [{unit}]')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')

    # Optional pressure contours
    if show_contours:
        # Build mesh of centers for contouring (slightly offset to cell centers)
        R, Z = np.meshgrid(r_c, z_c, indexing='ij')
        CS = plt.contour(r_c, z_c, (p/scale).T, levels=n_contours)
        plt.clabel(CS, inline=True, fontsize=8)

    # Free-surface overlay
    if z_free is not None:
        if np.isscalar(z_free):
            plt.hlines(float(z_free), r_edges[0], r_edges[-1], linewidth=2)
        elif callable(z_free):
            z_vals = z_free(r_c)
            plt.plot(r_c, z_vals, linewidth=2)
        else:
            z_vals = np.asarray(z_free)
            if z_free_is_edges:
                assert len(z_vals) == Nr+1, "z_free edges must have length Nr+1"
                # plot on edges directly
                plt.plot(r_edges, z_vals, linewidth=2)
            else:
                assert len(z_vals) == Nr, "z_free centers must have length Nr"
                plt.plot(r_c, z_vals, linewidth=2)

    # Air-core overlay
    if r_core is not None:
        if np.isscalar(r_core):
            rc = float(r_core)
            # Shade a vertical band 0 <= r <= rc across full z-range
            plt.fill_betweenx(z_edges, 0.0, np.clip(rc, r_edges[0], r_edges[-1]), alpha=0.2, hatch='//', edgecolor='k', linewidth=0.0)
            # Also draw the radius as a line
            plt.vlines(rc, z_edges[0], z_edges[-1], linewidth=2)
        elif callable(r_core):
            rc_vals = r_core(z_c)  # assume defined on z-centers
            plt.plot(rc_vals, z_c, linewidth=2)
        else:
            rc_vals = np.asarray(r_core)
            if r_core_is_edges:
                assert len(rc_vals) == Nz+1, "r_core edges must have length Nz+1"
                plt.plot(rc_vals, z_edges, linewidth=2)
            else:
                assert len(rc_vals) == Nz, "r_core centers must have length Nz"
                plt.plot(rc_vals, z_c, linewidth=2)

    if title: plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180)
    plt.show()
