"""
bc_segments_patch.py

This patch adds:
- Segment-aware boundary conditions on r=R and z-boundaries.
- Dirichlet segments for u_theta on r=R (swirl injector).
- An AUTO drain at the bottom (z=Zmin) that scales a user shape to balance side-wall inflow.
- Matching Neumann terms in the pressure-correction Poisson RHS.

See previous instructions for usage.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Any

Array = np.ndarray
from typing import Union
Profile = Union[float , Array , Callable[[Array], Array]]

# -----------------------
# Helpers
# -----------------------

def _mask_on_r_wall_segment(grid, z0: float, z1: float):
    zc = grid.z_c
    zmin, zmax = (z0, z1) if z0 <= z1 else (z1, z0)
    return (zc >= zmin) & (zc <= zmax)

def _mask_on_z_bottom_segment(grid, r0: float, r1: float):
    rc = grid.r_c
    rmin, rmax = (r0, r1) if r0 <= r1 else (r1, r0)
    return (rc >= rmin) & (rc <= rmax)

def _as_profile_z(values: Profile, z_centers: Array) -> Array:
    Nz = len(z_centers)
    if callable(values):
        arr = np.asarray(values(z_centers), dtype=float)
        assert arr.shape == (Nz,)
        return arr
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(Nz, float(arr))
    assert arr.shape == (Nz,)
    return arr

def _as_profile_r(values: Profile, r_centers: Array) -> Array:
    Nr = len(r_centers)
    if callable(values):
        arr = np.asarray(values(r_centers), dtype=float)
        assert arr.shape == (Nr,)
        return arr
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(Nr, float(arr))
    assert arr.shape == (Nr,)
    return arr

# -----------------------
# Boundary application (normals)
# -----------------------

def apply_boundary_normal_segmented(u_r: Array, u_z: Array, grid, bc: Dict[str, Any]):
    Nr, Nz = grid.Nr, grid.Nz
    r_c, z_c = grid.r_c, grid.z_c

    # defaults
    u_r[0, :]  = 0.0
    u_r[-1, :] = 0.0
    u_z[:, 0]  = 0.0
    u_z[:, -1] = 0.0

    # r=R segmented normal velocity
    rR = bc.get('r=R', {})
    if 'u_r' in rR:
        spec = rR['u_r']
        if isinstance(spec, list):
            for kind, payload in spec:
                if kind == 'segment':
                    z0, z1, prof = payload
                    m = _mask_on_r_wall_segment(grid, z0, z1)
                    val = _as_profile_z(prof, z_c)
                    u_r[-1, m] = val[m]
        elif isinstance(spec, tuple) and spec[0] in ('inlet','outlet'):
            val = _as_profile_z(spec[1], z_c)
            u_r[-1, :] = val

# -----------------------
# u_theta boundary (segments on r=R)
# -----------------------

def apply_ut_boundary_segmented(u_t: Array, grid, bc: Dict[str, Any]):
    rR = bc.get('r=R', {})
    z_c = grid.z_c

    # r=0 axis regularity
    u_t[0, :] = 0.0

    # r=R segments
    if 'ut' in rR:
        spec = rR['ut']
        if isinstance(spec, list):
            for kind, payload in spec:
                if kind == 'segment_dirichlet':
                    z0, z1, prof = payload
                    m = _mask_on_r_wall_segment(grid, z0, z1)
                    val = _as_profile_z(prof, z_c)
                    u_t[-1, m] = val[m]
        elif isinstance(spec, tuple) and spec[0] == 'dirichlet':
            val = _as_profile_z(spec[1], z_c)
            u_t[-1, :] = val

# -----------------------
# Poisson RHS with segmented Neumann terms
# -----------------------

def poisson_rhs_with_bc_segmented(grid, u_r_star: Array, u_z_star: Array, dt: float, rho: float,
                                  bc: Dict[str, Any], u_bottom: Array | None = None):
    Ae, Aw, An, As, V = grid.A_e, grid.A_w, grid.A_n, grid.A_s, grid.V
    div_u = (u_r_star[1:, :] * Ae - u_r_star[:-1, :] * Aw +
             u_z_star[:, 1:] * An - u_z_star[:, :-1] * As) / V
    rhs = -div_u / dt

    Nr, Nz = grid.Nr, grid.Nz
    z_c, r_c = grid.z_c, grid.r_c

    # r=R segments
    rR = bc.get('r=R', {})
    if 'u_r' in rR:
        spec = rR['u_r']
        if isinstance(spec, list):
            for kind, payload in spec:
                if kind == 'segment':
                    z0, z1, prof = payload
                    m = _mask_on_r_wall_segment(grid, z0, z1)
                    val = _as_profile_z(prof, z_c)
                    i = Nr-1
                    rhs[i, m] += (Ae[i, m] / (V[i, m] * dt)) * (u_r_star[i+1, m] - val[m])
        elif isinstance(spec, tuple) and spec[0] in ('inlet','outlet'):
            val = _as_profile_z(spec[1], z_c)
            i = Nr-1
            rhs[i, :] += (Ae[i, :] / (V[i, :] * dt)) * (u_r_star[i+1, :] - val[:])

    # AUTO drain Neumann term
    if u_bottom is not None:
        j = 0
        rhs[:, j] += (As[:, j] / (V[:, j] * dt)) * (u_z_star[:, j] - u_bottom)

    return rhs.ravel(order='F')

# -----------------------
# Global mass balance helpers
# -----------------------

def integrate_wall_inflow_rR(bc, grid) -> float:
    rR = bc.get('r=R', {})
    if 'u_r' not in rR:
        return 0.0
    zc = grid.z_c
    dz = grid.z_edges[1:] - grid.z_edges[:-1]
    R = grid.r_edges[-1]
    Q = 0.0
    spec = rR['u_r']
    if isinstance(spec, list):
        for kind, payload in spec:
            if kind == 'segment':
                z0, z1, prof = payload
                m = _mask_on_r_wall_segment(grid, z0, z1)
                val = _as_profile_z(prof, zc)
                Q += float(np.sum(val[m] * (2*np.pi*R) * dz[m]))
    elif isinstance(spec, tuple) and spec[0] in ('inlet','outlet'):
        val = _as_profile_z(spec[1], zc)
        Q += float(np.sum(val * (2*np.pi*R) * dz))
    return Q

def make_bottom_drain_profile_auto(bc, grid, Q_in: float):
    z0 = bc.get('z=Zmin', {})
    if 'u_z' not in z0:
        return None
    spec = z0['u_z']
    if not (isinstance(spec, tuple) and spec[0] == 'drain_auto'):
        return None
    payload = spec[1]
    rc = grid.r_c
    dr = grid.r_edges[1:] - grid.r_edges[:-1]
    f = _as_profile_r(payload['profile_r'], rc)
    r0, r1 = payload.get('region', (rc[0], grid.r_edges[-1]))
    # mask the region blocked by the wall, no drain there
    m = _mask_on_z_bottom_segment(grid, r0, grid.wall_rmin)
    denom = float(np.sum(2*np.pi * rc[m] * f[m] * dr[m]))
    if abs(denom) < 1e-15:
        return np.zeros_like(rc)
    s = Q_in / denom
    u_bottom = np.zeros_like(rc)
    u_bottom[m] = - s * f[m]
    return u_bottom
