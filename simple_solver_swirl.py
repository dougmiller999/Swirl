
"""
simple_solver_swirl.py — SIMPLE loop for an axisymmetric (r–z) staggered FV solver
including the azimuthal velocity equation u_theta (cell-centered), gravity, and inlet/outlet hooks.

Grid staggering:
  - p[i,j]        : cell-centered pressure, shape (Nr, Nz)
  - u_r[i_face,j] : radial face velocities at r-edges, shape (Nr+1, Nz)
  - u_z[i,j_face] : axial  face velocities at z-edges, shape (Nr, Nz+1)
  - u_t[i,j]      : cell-centered azimuthal velocity (u_theta), shape (Nr, Nz)

u_theta transport (axisymmetric, incompressible):
  ∂u_t/∂t + ∇·(u u_t) = ν ( ∇² u_t - u_t / r^2 )

Discretization notes:
  - Convection: upwind on faces using mass fluxes from current (u_r, u_z).
  - Diffusion: centered second-order using axisymmetric areas/volumes; -ν u_t/r^2 as a volumetric sink.
  - Pressure correction (SIMPLE): div((1/ρ) grad p') = -(1/Δt) div(u*) + boundary terms; u^{new} = u* - (Δt/ρ) grad p'.
  - Gravity in -z is applied in the predictor step for u_z.

Boundary hooks:
  bc = {
    'r=0':    {'u_r': ('wall', 0.0),         'ut': ('dirichlet', 0.0)},     # axis: u_r=0, u_t=0
    'r=R':    {'u_r': ('wall', 0.0),         'ut': ('dirichlet', profile_z)},
    'z=Zmin': {'u_z': ('inlet', profile_r),  'ut': ('dirichlet', profile_r) or ('neumann', 0.0)},
    'z=Zmax': {'u_z': ('outlet', profile_r), 'ut': ('dirichlet', profile_r) or ('neumann', 0.0)},
  }
  - For 'ut' on r=R (a vertical boundary), profile_z can be:
      scalar, array of length Nz (on z-centers), or callable f(z)->array(Nz).
  - For 'ut' on z-boundaries, profile_r is scalar, array of length Nr (on r-centers), or callable f(r)->array(Nr).
  - Defaults: u_r walls at r=0 and r=R; u_z walls at z=Zmin,Zmax unless specified;
              u_t: r=0 Dirichlet(0), r=R Dirichlet(0), z-boundaries Neumann(0).

CFL/stability: the explicit u_t step uses upwind convection + diffusion; choose Δt small enough for stability.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Callable, Any
from scipy.sparse.linalg import spsolve

from axisym_poisson_projection import (
    build_grid, assemble_poisson_matrix, divergence_from_face_fluxes,
    anchor_pressure
)

Array = np.ndarray
# Profile = float | Array | Callable[[Array], Array]
from typing import Union
Profile = Union[float , Array , Callable[[Array], Array]]

def _as_profile_r(values: Profile, r_centers: Array) -> Array:
    """Normalize a profile spec (scalar/array/callable) to an array on r-centers (len Nr)."""
    Nr = len(r_centers)
    if callable(values):
        arr = np.asarray(values(r_centers), dtype=float)
        assert arr.shape == (Nr,), "Callable profile must return length Nr."
        return arr
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(Nr, float(arr))
    assert arr.shape == (Nr,), "Array profile must be length Nr."
    return arr

def _as_profile_z(values: Profile, z_centers: Array) -> Array:
    """Normalize a profile spec (scalar/array/callable) to an array on z-centers (len Nz)."""
    Nz = len(z_centers)
    if callable(values):
        arr = np.asarray(values(z_centers), dtype=float)
        assert arr.shape == (Nz,), "Callable profile must return length Nz."
        return arr
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(Nz, float(arr))
    assert arr.shape == (Nz,), "Array profile must be length Nz."
    return arr

def _apply_boundary_normal(u_r: Array, u_z: Array, grid, bc: Dict[str, Dict[str, Tuple[str, Profile]]]):
    """Impose boundary normal velocities on the predictor/corrected fields."""
    Nr, Nz = grid.Nr, grid.Nz
    r_c = grid.r_c

    # Defaults: walls (no-penetration)
    r0_kind, r0_prof = bc.get('r=0',   {}).get('u_r', ('wall', 0.0))
    rR_kind, rR_prof = bc.get('r=R',   {}).get('u_r', ('wall', 0.0))
    z0_kind, z0_prof = bc.get('z=Zmin',{}).get('u_z', ('wall', 0.0))
    zN_kind, zN_prof = bc.get('z=Zmax',{}).get('u_z', ('wall', 0.0))

    if r0_kind in ('wall','symmetry'): u_r[0, :]  = 0.0
    if rR_kind in ('wall',):           u_r[-1, :] = 0.0
    if z0_kind in ('wall','symmetry'): u_z[:, 0]  = 0.0
    if zN_kind in ('wall',):           u_z[:, -1] = 0.0

    if z0_kind in ('inlet','outlet'):
        prof = _as_profile_r(z0_prof, r_c)
        u_z[:, 0] = prof
    if zN_kind in ('inlet','outlet'):
        prof = _as_profile_r(zN_prof, r_c)
        u_z[:, -1] = prof

    if rR_kind in ('inlet','outlet'):
        if np.ndim(rR_prof)==0:
            u_r[-1, :] = float(rR_prof)
        else:
            raise ValueError("For r=R inlet/outlet, provide a scalar normal velocity for now.")

def _poisson_rhs_with_bc(grid, u_r_star: Array, u_z_star: Array, dt: float, rho: float,
                         bc: Dict[str, Dict[str, Tuple[str, Profile]]]) -> Array:
    """RHS for pressure-correction Poisson including Neumann terms for inlet/outlet targets."""
    Nr, Nz = grid.Nr, grid.Nz
    A_e, A_w, A_n, A_s, V = grid.A_e, grid.A_w, grid.A_n, grid.A_s, grid.V
    r_c = grid.r_c
    rhs = -(divergence_from_face_fluxes(u_r_star, u_z_star, grid).reshape(Nr, Nz)) / dt

    z0_kind, z0_prof = bc.get('z=Zmin',{}).get('u_z', ('wall', 0.0))
    zN_kind, zN_prof = bc.get('z=Zmax',{}).get('u_z', ('wall', 0.0))
    rR_kind, rR_prof = bc.get('r=R',   {}).get('u_r', ('wall', 0.0))

    if z0_kind in ('inlet','outlet'):
        prof = _as_profile_r(z0_prof, r_c)
        for i in range(Nr):
            rhs[i,0] += (A_s[i,0] / (V[i,0] * dt)) * (u_z_star[i,0] - prof[i])
    if zN_kind in ('inlet','outlet'):
        prof = _as_profile_r(zN_prof, r_c)
        j = Nz-1
        for i in range(Nr):
            rhs[i,j] += (A_n[i,j] / (V[i,j] * dt)) * (u_z_star[i,j+1] - prof[i])
    if rR_kind in ('inlet','outlet'):
        i = Nr-1
        for j in range(Nz):
            rhs[i,j] += (A_e[i,j] / (V[i,j] * dt)) * (u_r_star[i+1,j] - float(rR_prof))

    return rhs.flatten(order='F')

def _apply_ut_boundary(u_t: Array, grid, bc):
    """Apply u_theta boundary conditions (Dirichlet or Neumann)."""
    Nr, Nz = grid.Nr, grid.Nz
    r_c, z_c = grid.r_c, grid.z_c

    # Defaults
    r0_kind, r0_prof = bc.get('r=0',   {}).get('ut', ('dirichlet', 0.0))
    rR_kind, rR_prof = bc.get('r=R',   {}).get('ut', ('dirichlet', 0.0))
    z0_kind, z0_prof = bc.get('z=Zmin',{}).get('ut', ('neumann', 0.0))
    zN_kind, zN_prof = bc.get('z=Zmax',{}).get('ut', ('neumann', 0.0))

    # r=0 (axis): Dirichlet 0 by default (regularity)
    if r0_kind == 'dirichlet':
        u_t[0,:] = 0.0
    # r=R
    if rR_kind == 'dirichlet':
        prof = _as_profile_z(rR_prof, z_c)
        u_t[-1,:] = prof
    elif rR_kind == 'neumann':
        # zero-gradient: copy neighbor
        u_t[-1,:] = u_t[-2,:]

    # z=Zmin
    if z0_kind == 'dirichlet':
        prof = _as_profile_r(z0_prof, r_c)
        u_t[:,0] = prof
    elif z0_kind == 'neumann':
        u_t[:,0] = u_t[:,1]
    # z=Zmax
    if zN_kind == 'dirichlet':
        prof = _as_profile_r(zN_prof, r_c)
        u_t[:,-1] = prof
    elif zN_kind == 'neumann':
        u_t[:,-1] = u_t[:,-2]

def _update_utheta(u_t: Array, u_r: Array, u_z: Array, grid, nu: float, dt: float, bc) -> Array:
    """One explicit Euler step for u_theta with upwind convection and diffusion (axisymmetric)."""
    Nr, Nz = grid.Nr, grid.Nz
    u_new = u_t.copy()

    r_c = grid.r_c; z_c = grid.z_c
    V   = grid.V; A_e, A_w, A_n, A_s = grid.A_e, grid.A_w, grid.A_n, grid.A_s

    # center-to-center distances (uniform if grid was built linear)
    dr_e = np.empty(Nr); dr_e[:-1] = r_c[1:] - r_c[:-1]; dr_e[-1] = np.nan
    dr_w = np.empty(Nr); dr_w[1:]  = r_c[1:] - r_c[:-1]; dr_w[0]  = np.nan
    dz_n = np.empty(Nz); dz_n[:-1] = z_c[1:] - z_c[:-1]; dz_n[-1] = np.nan
    dz_s = np.empty(Nz); dz_s[1:]  = z_c[1:] - z_c[:-1]; dz_s[0]  = np.nan

    # Upwind helper
    def upwind(phiP, phiN, flux):
        return phiP if flux >= 0.0 else phiN

    for j in range(Nz):
        for i in range(Nr):
            # Mass fluxes on faces
            Phi_e = u_r[i+1, j] * A_e[i, j]  # east face
            Phi_w = u_r[i,   j] * A_w[i, j]  # west face
            Phi_n = u_z[i, j+1] * A_n[i, j]  # north face
            Phi_s = u_z[i, j  ] * A_s[i, j]  # south face

            # Neighbor values with clamping at boundaries (ghost via copy)
            uP = u_t[i,j]
            uE = u_t[i+1,j] if i < Nr-1 else uP
            uW = u_t[i-1,j] if i > 0     else uP
            uN = u_t[i,j+1] if j < Nz-1 else uP
            uS = u_t[i,j-1] if j > 0     else uP

            # Upwind face values
            u_e = upwind(uP, uE, Phi_e)
            u_w = upwind(uW, uP, Phi_w)
            u_n = upwind(uP, uN, Phi_n)
            u_s = upwind(uS, uP, Phi_s)

            # Convective divergence
            conv = (Phi_e*u_e - Phi_w*u_w + Phi_n*u_n - Phi_s*u_s) / V[i,j]

            # Diffusion fluxes (centered)
            diff_e = nu * A_e[i,j] * ( (uE - uP) / dr_e[i] ) if i < Nr-1 else 0.0
            diff_w = nu * A_w[i,j] * ( (uP - uW) / dr_w[i] ) if i > 0     else 0.0
            diff_n = nu * A_n[i,j] * ( (uN - uP) / dz_n[j] ) if j < Nz-1 else 0.0
            diff_s = nu * A_s[i,j] * ( (uP - uS) / dz_s[j] ) if j > 0     else 0.0
            diff = (diff_e - diff_w + diff_n - diff_s) / V[i,j]

            # Axisymmetric sink term -nu * u_t / r^2 (avoid r=0 by enforcing u_t=0 there)
            sink = - nu * (uP / (r_c[i]**2)) if r_c[i] > 0.0 else 0.0

            # Explicit update: ∂u_t/∂t = -conv + diff + sink
            u_new[i,j] = uP + dt * (-conv + diff + sink)

    # Apply u_theta boundary conditions
    _apply_ut_boundary(u_new, grid, bc)
    return u_new

def grad_p_on_faces(p: Array, r_c: Array, z_c: Array):
    """Pressure gradients to faces (interior) for correction/predictor steps."""
    Nr, Nz = p.shape
    dpdr = np.zeros((Nr+1, Nz))
    for j in range(Nz):
        for i_face in range(1, Nr):
            iW = i_face - 1; iE = i_face
            dr_face = r_c[iE] - r_c[iW]
            dpdr[i_face, j] = (p[iE, j] - p[iW, j]) / dr_face
    dpdz = np.zeros((Nr, Nz+1))
    for i in range(Nr):
        for j_face in range(1, Nz):
            jS = j_face - 1; jN = j_face
            dz_face = z_c[jN] - z_c[jS]
            dpdz[i, j_face] = (p[i, jN] - p[i, jS]) / dz_face
    return dpdr, dpdz

def laplacian_faces(u_face: Array, dr: float, dz: float, orient='r'):
    """Simple 5-pt Laplacian on face grids (uniform spacing)."""
    if orient == 'r':
        Nr_p1, Nz = u_face.shape
        Nr = Nr_p1 - 1
        lap = np.zeros_like(u_face)
        for j in range(Nz):
            for i_face in range(1, Nr):
                lap[i_face, j] = (
                    (u_face[i_face+1, j] - 2*u_face[i_face, j] + u_face[i_face-1, j])/(dr*dr) +
                    ( (u_face[i_face, j-1] if j>0 else 0.0) - 2*u_face[i_face, j] + (u_face[i_face, j+1] if j<Nz-1 else 0.0) )/(dz*dz)
                )
        return lap
    else:
        Nr, Nz_p1 = u_face.shape
        Nz = Nz_p1 - 1
        lap = np.zeros_like(u_face)
        for i in range(Nr):
            for j_face in range(1, Nz):
                lap[i, j_face] = (
                    ( (u_face[i-1, j_face] if i>0 else 0.0) - 2*u_face[i, j_face] + (u_face[i+1, j_face] if i<Nr-1 else 0.0) )/(dr*dr) +
                    (u_face[i, j_face+1] - 2*u_face[i, j_face] + u_face[i, j_face-1])/(dz*dz)
                )
        return lap

def simple_solve_swirl(Nr=40, Nz=40, R=0.1, H=5.0, Zmin=0.0, Zmax=0.3,
                       rho=1000.0, mu=0.0, g=9.81,
                       dt=0.01, max_iter=500, alpha_p=0.3,
                       tol_div=1e-8, verbose=True,
                       bc: Dict[str, Dict[str, Tuple[str, Profile]]] | None = None,
                       u_t_init: Array | None = None):
    """SIMPLE loop with u_theta transport. Returns fields and history."""
    if bc is None: bc = {}
    grid = build_grid(Nr, Nz, R, Zmin, Zmax)
    dr = grid.dr.mean(); dz = grid.dz.mean()
    nu = mu / rho if mu>0 else 0.0

    # Fields
    p = np.zeros((Nr, Nz))
    u_r = np.zeros((Nr+1, Nz))
    u_z = np.zeros((Nr,   Nz+1))
    u_t = np.zeros((Nr, Nz)) if u_t_init is None else u_t_init.copy()

    # Apply initial u_t BCs
    _apply_ut_boundary(u_t, grid, bc)

    # Poisson operator (constant ρ)
    Ap = assemble_poisson_matrix(grid, rho)

    history = []
    for it in range(1, max_iter+1):
        # Predictor for meridional velocities
        dpdr, dpdz = grad_p_on_faces(p, grid.r_c, grid.z_c)
        u_r_star = u_r - (dt/rho)*dpdr
        u_z_star = u_z - (dt/rho)*dpdz - dt*g

        # --- NEW: add centrifugal acceleration from swirl on radial faces ---
        # face radii r_f = r_edges[1:-1] for interior faces
        r_f = grid.r_edges[1:-1]                   # shape (Nr-1,)
        # interpolate u_theta from centers to the two cells adjacent to each radial face
        # then average to the face; result shape (Nr-1, Nz)
        u_t_left  = u_t[:-1, :]                    # (Nr-1, Nz)
        u_t_right = u_t[1:,  :]                    # (Nr-1, Nz)
        u_t_face  = 0.5*(u_t_left + u_t_right)
        # a_c = u_theta^2 / r at the face
        a_c = (u_t_face**2) / r_f[:, None]         # broadcast r_f along z
        # apply to interior radial faces; boundaries remain whatever BCs enforce
        u_r_star[1:-1, :] += dt * a_c
        # --- end NEW ---
        
        if nu > 0.0:
            u_r_star += dt * nu * laplacian_faces(u_r, dr, dz, orient='r')
            u_z_star += dt * nu * laplacian_faces(u_z, dr, dz, orient='z')

        # Apply boundary normal velocities
        _apply_boundary_normal(u_r_star, u_z_star, grid, bc)

        # Pressure-correction Poisson
        b = _poisson_rhs_with_bc(grid, u_r_star, u_z_star, dt, rho, bc)
        Ap_anch, b_anch = anchor_pressure(Ap.copy(), b.copy(), idx=0)
        p_prime = spsolve(Ap_anch, b_anch).reshape((Nr, Nz), order='F')

        # Update pressure and correct velocities
        oldP = p.copy()
        p += alpha_p * p_prime
        dpdr_p, dpdz_p = grad_p_on_faces(p_prime, grid.r_c, grid.z_c)
        # dpdr_p, dpdz_p = grad_p_on_faces(alpha_p*p_prime, grid.r_c, grid.z_c)
        u_r = u_r_star - (dt/rho) * dpdr_p
        u_z = u_z_star - (dt/rho) * dpdz_p

        # Re-enforce boundary normals
        _apply_boundary_normal(u_r, u_z, grid, bc)

        # Update u_theta explicitly with current (u_r,u_z)
        u_t = _update_utheta(u_t, u_r, u_z, grid, nu, dt, bc)

        # Diagnostics
        div_u = divergence_from_face_fluxes(u_r, u_z, grid)
        max_div = np.max(np.abs(div_u))
        res_p = np.max(np.abs(p-oldP))
        # Track u_t change (Linf)
        hist_ut = np.max(np.abs(_update_utheta(u_t, u_r, u_z, grid, nu, 0.0, bc) - u_t))  # zero dt -> just BC reapply
        history.append((it, max_div, res_p, hist_ut))

        if verbose and (it % 20 == 0 or it == 1):
            print(f"Iter {it:4d}: max(div u)={max_div:.3e}  max|p'-oldP|={res_p:.3e}")

        if (it % 1000 == 0 or it == 1):
            # -------------------
            # Plot: pressure colormap + u_theta contours
            # -------------------
            # adjust to keep mass conservation
            V_target = np.pi * (grid.R**2) * H          # your known fill volume
            c = choose_pressure_offset_for_volume(p, grid, V_target, patm=0.0)
            p += c # now the P=0 contour will contain our mass
            plot_pressure_contours(p, grid,
                                   title='Rotating bucket: pressure contours %d iters' % (it),
                                   unit='bar', scale=1e5, n_contours=10)
            

        # if max_div < tol_div and res_p < 1e-8:
        if max_div < tol_div and res_p < 10:
            if verbose:
                print(f"Converged (continuity & pressure) at iter {it}: max(div u)={max_div:.3e}")
            break

    return {'p': p, 'u_r': u_r, 'u_z': u_z, 'u_t': u_t, 'grid': grid, 'history': history}

import matplotlib.pyplot as plt
import numpy as np

def _cell_center_velocities(u_r, u_z, grid):
    '''Compute cell-centered (u_r,u_z) by averaging adjacent face values.'''
    Nr, Nz = grid.Nr, grid.Nz
    ur_c = 0.5*(u_r[0:Nr, :] + u_r[1:Nr+1, :])      # average in r
    uz_c = 0.5*(u_z[:, 0:Nz] + u_z[:, 1:Nz+1])      # average in z
    return ur_c, uz_c

def plot_pressure_contours(p, grid, title=None, unit='Pa', scale=1.0, n_contours=16):
    '''Pressure contours on (r,z) grid.'''
    Nr, Nz = p.shape
    r_edges, z_edges = grid.r_edges, grid.z_edges
    r_c, z_c = grid.r_c, grid.z_c

    plt.figure(figsize=(7,5))
    pm = plt.pcolormesh(r_edges, z_edges, (p.T)/scale, shading='auto')
    cbar = plt.colorbar(pm); cbar.set_label(f'Pressure [{unit}]')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    CS = plt.contour(r_c, z_c, (p.T)/scale, levels=n_contours, colors='white')
    plt.clabel(CS, inline=True, fontsize=10)
    if title: plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pressure_with_swirl(p, u_t, grid, title=None, unit='Pa', scale=1.0, show_contours=True, n_contours=16):
    '''Pressure colormap on (r,z) with optional u_theta contours on cell centers.'''
    Nr, Nz = p.shape
    r_edges, z_edges = grid.r_edges, grid.z_edges
    r_c, z_c = grid.r_c, grid.z_c

    plt.figure(figsize=(7,5))
    pm = plt.pcolormesh(r_edges, z_edges, (p.T)/scale, shading='auto')
    cbar = plt.colorbar(pm); cbar.set_label(f'Pressure [{unit}]')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    if show_contours:
        # Contour u_theta on centers
        CS = plt.contour(r_c, z_c, u_t.T, levels=n_contours)
        plt.clabel(CS, inline=True, fontsize=8)
    if title: plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_streamlines(u_r, u_z, grid, density=1.5, title='Streamlines (u_r, u_z)'):
    '''Streamlines of the meridional velocity field on cell centers.'''
    r_c, z_c = grid.r_c, grid.z_c
    ur_c, uz_c = _cell_center_velocities(u_r, u_z, grid)

    plt.figure(figsize=(7,5))
    # streamplot expects arrays in (x,y) order; our center fields are (Nr,Nz) corresponding to (r,z)
    plt.streamplot(r_c, z_c, ur_c.T, uz_c.T, density=density)
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

import numpy as np

def volume_from_pressure_isobar(p, grid, patm=0.0, offset=0.0):
    # Extract z_fs for p+offset = patm
    r_c, z_c = grid.r_c, grid.z_c
    Nr, Nz = p.shape
    z_fs = np.zeros(Nr)
    for i in range(Nr):
        col = p[i,:] + offset - patm
        # default: surface at top if top cell >= 0
        if col[-1] >= 0:
            z_fs[i] = grid.z_edges[-1]
            # print("   col[-1] = %12.5e"%col[-1],": i = ", i, "z_fs[i]= ",z_fs[i])
            continue
        # find first sign change from top down
        found = False
        for j in range(Nz-1, 0, -1):
            if (col[j] <= 0 < col[j-1]) or (col[j] < 0 <= col[j-1]):
                z1, z0 = z_c[j], z_c[j-1]
                p1, p0 = col[j], col[j-1]
                t = 0.0 if p1==p0 else (-p0)/(p1-p0)
                z_fs[i] = np.clip(z0 + t*(z1-z0), grid.z_edges[0], grid.z_edges[-1])
                found = True
                break
        if not found:
            # whole column positive → surface below bottom (empty column): set to bottom
            z_fs[i] = grid.z_edges[0]
        # print("   Found=",found, " i = ", i, "z_fs[i]= ",z_fs[i])
    # Axisymmetric volume: 2π ∫ r z(r) dr  ≈ 2π Σ r_c z_fs Δr
    dr = grid.r_edges[1:] - grid.r_edges[:-1]
    return 2*np.pi * np.sum(r_c * z_fs * dr)

def choose_pressure_offset_for_volume(p, grid, V_target, patm=0.0, c_lo=-1e7, c_hi=+1e7, tol=1e-6):
    # Bisection on offset c so that volume_from_pressure_isobar(p,grid,patm,c)=V_target
    f_lo = volume_from_pressure_isobar(p, grid, patm, c_lo) - V_target
    f_hi = volume_from_pressure_isobar(p, grid, patm, c_hi) - V_target
    if f_lo * f_hi > 0:
        # fallback: return zero shift if not bracketed
        return 0.0
    for _ in range(60):
        c_mid = 0.5*(c_lo + c_hi)
        f_mid = volume_from_pressure_isobar(p, grid, patm, c_mid) - V_target
        # print(f"{_:3d} f_mid volume = {f_mid:.3e}" )
        # print(" c_mid = ", c_mid)
        # print(" volume_from_pressure_isobar(p,grid,patm,c_mid) = ", volume_from_pressure_isobar(p,grid,patm,c_mid))
        if abs(f_mid) <= max(1.0, abs(V_target))*tol:
            return c_mid
        if f_lo * f_mid <= 0:
            c_hi, f_hi = c_mid, f_mid
        else:
            c_lo, f_lo = c_mid, f_mid
    return 0.5*(c_lo + c_hi)
    
if __name__ == "__main__":
    # Demo: rotating outer wall (Ω) with no inlet/outlet; swirl diffuses inward; hydrostatic in z.
    Nr, Nz = 60, 60
    R, Zmin, Zmax = 0.1, 0.0, 0.2
    rho, mu, g = 1000.0, 1e-3, 9.81
    dt = 0.001

    Omega = 50.0   # rad/s
    bc = {
        'r=0':    {'u_r': ('wall', 0.0), 'ut': ('dirichlet', 0.0)},
        'r=R':    {'u_r': ('wall', 0.0), 'ut': ('dirichlet', lambda z: Omega*R*np.ones_like(z))},
        'z=Zmin': {'u_z': ('wall', 0.0), 'ut': ('neumann', 0.0)},
        'z=Zmax': {'u_z': ('wall', 0.0), 'ut': ('neumann', 0.0)},
    }

    out = simple_solve_swirl(Nr=Nr, Nz=Nz, R=R, Zmin=Zmin, Zmax=Zmax,
                             rho=rho, mu=mu, g=g,
                             dt=dt, max_iter=200, alpha_p=0.7, tol_div=1e-8, verbose=True,
                             bc=bc)

    p = out['p']; u_t = out['u_t']; grid = out['grid']
    print("Done. u_theta range [m/s]:", float(u_t.min()), "to", float(u_t.max()))

    
