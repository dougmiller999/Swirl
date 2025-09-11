
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
from scipy import sparse

from graphics import (
    plot_pressure_contours, 
    plot_pressure_with_swirl,
    plot_streamlines,
    choose_pressure_offset_for_volume
)

from axisym_poisson_projection import (
    build_grid, assemble_poisson_matrix, divergence_from_face_fluxes,
    anchor_pressure
)

from bc_segments_patch import (
    apply_boundary_normal_segmented,
    apply_ut_boundary_segmented,
    poisson_rhs_with_bc_segmented,
    integrate_wall_inflow_rR,
    make_bottom_drain_profile_auto,
)

from internal_wall_mask_patch import (
    build_slanted_wall_masks,
    apply_internal_wall_bc,
    divergence_from_face_fluxes_masked,
    poisson_rhs_with_bc_segmented_masked,
    r_wall_of_z,
)

from kinematic_eta import kinematic_eta_update

Array = np.ndarray
# Profile = float | Array | Callable[[Array], Array]
from typing import Union
Profile = Union[float , Array , Callable[[Array], Array]]

def _mask_on_r_wall_segment(grid, z0, z1):
    zc = grid.z_c  # (Nz,)
    return (zc >= min(z0,z1)) & (zc <= max(z0,z1))  # (Nz,)

def _mask_on_z_bottom_segment(grid, r0, r1):
    rc = grid.r_c  # (Nr,)
    return (rc >= min(r0,r1)) & (rc <= max(r0,r1))  # (Nr,)

def _as_profile_r(values: Profile, r_centers: Array) -> Array:
    """Normalize a profile spec (scalar/array/callable) to an array on r-centers (len Nr)."""
    if callable(values): arr = np.asarray(values(r_centers), float)
    else:
        arr = np.asarray(values, float)
        if arr.ndim == 0: arr = np.full_like(r_centers, float(values))
    assert arr.shape == (len(r_centers),)
    return arr

def _as_profile_z(values: Profile, z_centers: Array) -> Array:
    """Normalize a profile spec (scalar/array/callable) to an array on z-centers (len Nz)."""
    if callable(values):
        arr = np.asarray(values(z_centers), dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(z_centers), float(arr))
    assert arr.shape == (len(z_centers),), "Array profile must be length Nz."
    return arr

def _apply_boundary_normal(u_r: Array, u_z: Array, grid, bc: Dict[str, Dict[str, Tuple[str, Profile]]]):
    """Impose boundary normal velocities on the predictor/corrected fields."""
    Nr, Nz = grid.Nr, grid.Nz
    r_c = grid.r_c

    # Defaults: walls (no-penetration)
    u_r[0,:]  = 0.0  # r=0 axis
    u_r[-1,:] = 0.0  # r=R wall (may be overwritten by segments)
    u_z[:,0]  = 0.0  # z=Zmin wall (may be overwritten by drain)
    u_z[:,-1] = 0.0  # z=Zmax wall (may be overwritten)

     # r=R segmented inlet/outlet
    rR = bc.get('r=R', {})
    if 'u_r' in rR:
        spec = rR['u_r']
        if isinstance(spec, list):
            for kind, payload in spec:
                if kind == 'segment':
                    z0, z1, prof = payload
                    m = _mask_on_r_wall_segment(grid, z0, z1)     # (Nz,)
                    val = _as_profile_z(prof, z_c)                 # (Nz,)
                    u_r[-1, m] = val[m]
        elif isinstance(spec, tuple) and spec[0] in ('inlet','outlet'):
            # backward compat: whole-edge profile
            val = _as_profile_z(spec[1], z_c)
            u_r[-1,:] = val
   
    # z=Zmin bottom drain or segments
    z0 = bc.get('z=Zmin', {})
    if 'u_z' in z0:
        kind, payload = z0['u_z'] if isinstance(z0['u_z'], tuple) else (None, None)
        if kind == 'drain_auto':
            pass  # handled after we compute total inflow (see section C)
        elif kind == 'segment':
            r0, r1, prof = payload
            m = _mask_on_z_bottom_segment(grid, r0, r1)  # (Nr,)
            val = _as_profile_r(prof, r_c)
            u_z[m, 0] = val[m]
        elif kind in ('inlet','outlet'):  # whole-edge
            val = _as_profile_r(payload, r_c)
            u_z[:,0] = val

def _poisson_rhs_with_bc(grid, u_r_star, u_z_star, dt, rho, bc):
    """RHS for pressure-correction Poisson including Neumann terms for inlet/outlet targets."""
    Nr, Nz = grid.Nr, grid.Nz
    rhs = -(divergence_from_face_fluxes(u_r_star, u_z_star, grid)) / dt  # (Nr,Nz)
    # r=R segments → east faces of column i=Nr-1
    rR = bc.get('r=R', {})
    if 'u_r' in rR:
        spec = rR['u_r']
        if isinstance(spec, list):
            for kind, payload in spec:
                if kind == 'segment':
                    z0, z1, prof = payload
                    m = _mask_on_r_wall_segment(grid, z0, z1)           # (Nz,)
                    val = _as_profile_z(prof, grid.z_c)                  # (Nz,)
                    i = Nr-1
                    rhs[i, m] += (grid.A_e[i, m] / (grid.V[i, m] * dt)) * (u_r_star[i+1, m] - val[m])
        elif isinstance(spec, tuple) and spec[0] in ('inlet','outlet'):
            val = _as_profile_z(spec[1], grid.z_c)
            i = Nr-1
            rhs[i, :] += (grid.A_e[i, :] / (grid.V[i, :] * dt)) * (u_r_star[i+1, :] - val[:])

    # z=Zmin segments / drain handled similarly when known (see section C)
    # ...
    return rhs.ravel(order='F')

def integrate_wall_inflow_rR(bc, grid):
    """Compute total inflow (positive) prescribed on r=R from bc segments."""
    rR = bc.get('r=R', {})
    if 'u_r' not in rR: return 0.0
    zc = grid.z_c; dz = grid.z_edges[1:] - grid.z_edges[:-1]
    Q = 0.0
    R = grid.r_edges[-1]
    if isinstance(rR['u_r'], list):
        for kind, payload in rR['u_r']:
            if kind == 'segment':
                z0, z1, prof = payload
                m = _mask_on_r_wall_segment(grid, z0, z1)
                val = _as_profile_z(prof, zc)
                # positive val means inflow *into* domain
                Q += np.sum((val[m]) * (2*np.pi*R) * dz[m])
    elif isinstance(rR['u_r'], tuple) and rR['u_r'][0] in ('inlet','outlet'):
        val = _as_profile_z(rR['u_r'][1], zc)
        Q += np.sum(val * (2*np.pi*R) * dz)
    return float(Q)

def make_bottom_drain_profile_auto(bc, grid, Q_in):
    """Return u_z_bottom (Nr,) so that ∫ 2π r u_z dr = -Q_in over specified region."""
    z0 = bc.get('z=Zmin', {})
    if 'u_z' not in z0:  # nothing to do
        return None
    kind, payload = z0['u_z']
    if kind != 'drain_auto':
        return None
    rc = grid.r_c; dr = grid.r_edges[1:] - grid.r_edges[:-1]
    r0, r1 = payload.get('region', (rc[0], rc[-1]))
    f = _as_profile_r(payload['profile_r'], rc)  # shape (Nr,)
    m = _mask_on_z_bottom_segment(grid, r0, r1)
    denom = np.sum(2*np.pi*rc[m]*f[m]*dr[m])
    if abs(denom) < 1e-12:
        return np.zeros_like(rc)
    s = Q_in / denom
    u_bottom = np.zeros_like(rc)
    u_bottom[m] = - s * f[m]
    return u_bottom

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
    """
    Vectorized explicit Euler update for u_theta with upwind advection and
    centered diffusion (axisymmetric), plus -nu * u_t / r^2 sink.
    Shapes:
      u_t: (Nr, Nz)              [cell centers]
      u_r: (Nr+1, Nz)            [radial faces]
      u_z: (Nr,   Nz+1)          [axial faces]
    """
    Nr, Nz = u_t.shape
    r_c, z_c = grid.r_c, grid.z_c
    V   = grid.V
    A_e, A_w, A_n, A_s = grid.A_e, grid.A_w, grid.A_n, grid.A_s

    # --- Build neighbor fields (ghost via copy) ---
    uP = u_t
    uE = np.empty_like(u_t); uE[:-1, :] = u_t[1:, :];  uE[-1, :] = u_t[-1, :]
    uW = np.empty_like(u_t); uW[1:,  :] = u_t[:-1, :]; uW[0,  :] = u_t[0,  :]
    uN = np.empty_like(u_t); uN[:, :-1] = u_t[:, 1:];  uN[:, -1] = u_t[:, -1]
    uS = np.empty_like(u_t); uS[:, 1: ] = u_t[:, :-1]; uS[:,  0] = u_t[:,  0]

    # --- Face mass fluxes (already include axisymmetric areas) ---
    Phi_e = u_r[1:,  :] * A_e        # (Nr, Nz) east
    Phi_w = u_r[:-1, :] * A_w        # (Nr, Nz) west
    Phi_n = u_z[:, 1:] * A_n         # (Nr, Nz) north
    Phi_s = u_z[:, :-1] * A_s        # (Nr, Nz) south

    # --- Upwind values on faces (vectorized) ---
    u_e = np.where(Phi_e >= 0.0, uP, uE)
    u_w = np.where(Phi_w >= 0.0, uW, uP)
    u_n = np.where(Phi_n >= 0.0, uP, uN)
    u_s = np.where(Phi_s >= 0.0, uS, uP)

    # --- Convective divergence ---
    conv = (Phi_e * u_e - Phi_w * u_w + Phi_n * u_n - Phi_s * u_s) / V

    # --- Diffusion (centered) ---
    grad_e = (uE - uP) / grid.dr_e;   grad_e[-1, :] = 0.0    # no east neighbor at i=Nr-1
    grad_w = (uP - uW) / grid.dr_w;   grad_w[0,  :] = 0.0    # no west neighbor at i=0
    grad_n = (uN - uP) / grid.dz_n;   grad_n[:, -1] = 0.0    # no north neighbor at j=Nz-1
    grad_s = (uP - uS) / grid.dz_s;   grad_s[:,  0] = 0.0    # no south neighbor at j=0

    diff_e = nu * A_e * grad_e
    diff_w = nu * A_w * grad_w
    diff_n = nu * A_n * grad_n
    diff_s = nu * A_s * grad_s

    diff = (diff_e - diff_w + diff_n - diff_s) / V

    # --- Axisymmetric sink ---
    sink = -nu * uP * grid.inv_r2   # broadcasts over z

    # --- Explicit update ---
    u_new = uP + dt * (-conv + diff + sink)

    # Enforce u_theta boundary conditions (Dirichlet/Neumann)
    apply_ut_boundary_segmented(u_new, grid, bc)
    return u_new

def grad_p_on_faces(p, grid):
    """
    Vectorized pressure gradients to faces on the staggered axisymmetric grid.

    Parameters
    ----------
    p : (Nr, Nz)        cell-centered pressure
    grid : provides r_c, z_c; optionally dr_e (Nr,1), dz_n (1,Nz)

    Returns
    -------
    dpdr : (Nr+1, Nz)   radial-face grad p (interior faces filled, boundaries 0)
    dpdz : (Nr, Nz+1)   axial-face  grad p (interior faces filled, boundaries 0)
    """
    Nr, Nz = p.shape

    # --- radial faces (Nr+1, Nz) ---
    dpdr = np.zeros((Nr+1, Nz), dtype=p.dtype)
    # spacings at interior faces (i_face = 1..Nr-1)
    dr_face = grid.dr_e[:-1, 0]          # (Nr-1,)
    num_r = p[1:, :] - p[:-1, :]             # (Nr-1, Nz)
    dpdr[1:Nr, :] = num_r / dr_face[:, None] # broadcast over z

    # --- axial faces (Nr, Nz+1) ---
    dpdz = np.zeros((Nr, Nz+1), dtype=p.dtype)
    dz_face = grid.dz_n[0, :-1]          # (Nz-1,)
    num_z = p[:, 1:] - p[:, :-1]             # (Nr, Nz-1)
    dpdz[:, 1:Nz] = num_z / dz_face[None, :] # broadcast over r

    return dpdr, dpdz

def laplacian_faces(u_face, dr, dz, orient='r'):
    """
    Vectorized Laplacian on face grids (uniform spacings).
    - orient='r': u_face shape (Nr+1, Nz), fills interior i=1..Nr-1
    - orient='z': u_face shape (Nr,   Nz+1), fills interior j=1..Nz-1
    Homogeneous Dirichlet outside the domain (zero padding), same as your loop version.
    """
    if orient == 'r':
        Nr_p1, Nz = u_face.shape
        # r-second-derivative on interior radial faces
        d2r = (u_face[2:, :] - 2.0*u_face[1:-1, :] + u_face[:-2, :]) / (dr*dr)
        # z-second-derivative with zero padding at top/bottom
        u_pad = np.pad(u_face[1:-1, :], ((0,0),(1,1)), mode='constant', constant_values=0.0)
        d2z = (u_pad[:, 2:] - 2.0*u_pad[:, 1:-1] + u_pad[:, :-2]) / (dz*dz)
        lap = np.zeros_like(u_face)
        lap[1:-1, :] = d2r + d2z
        return lap
    else:
        Nr, Nz_p1 = u_face.shape
        # z-second-derivative on interior axial faces
        d2z = (u_face[:, 2:] - 2.0*u_face[:, 1:-1] + u_face[:, :-2]) / (dz*dz)
        # r-second-derivative with zero padding at axis/wall
        u_pad = np.pad(u_face[:, 1:-1], ((1,1),(0,0)), mode='constant', constant_values=0.0)
        d2r = (u_pad[2:, :] - 2.0*u_pad[1:-1, :] + u_pad[:-2, :]) / (dr*dr)
        lap = np.zeros_like(u_face)
        lap[:, 1:-1] = d2r + d2z
        return lap


def _p_dof_index(i, j, Nr):
    """Fortran-style (i,j) -> k mapping for p vectorized with order='F'."""
    return int(i + j * Nr)

def anchor_pressure_exact(A_csr, b, k, value=0.0, keep_spd=True):
    """Exact Dirichlet at DOF k (CSR/CSC only, no LIL)."""
    A = A_csr.tocsr(copy=True)

    # zero ROW k, set diag to 1
    rs, re = A.indptr[k], A.indptr[k+1]
    A.data[rs:re] = 0.0
    cols = A.indices[rs:re]
    hit = np.where(cols == k)[0]
    if hit.size:
        A.data[rs + hit[0]] = 1.0
    else:
        A += sparse.csr_matrix(([1.0], ([k], [k])), shape=A.shape)
    b[k] = value

    if keep_spd:
        # zero COLUMN k except diagonal
        Ac = A.tocsc()
        cs, ce = Ac.indptr[k], Ac.indptr[k+1]
        rows = Ac.indices[cs:ce]
        mask = rows != k
        Ac.data[cs:ce][mask] = 0.0
        A = Ac.tocsr()

    return A, b

def anchor_pressure_exact_many(A_csr, b, idx, values=None, keep_spd=True):
    """Exact Dirichlet at many DOFs (one CSC pass)."""
    A = A_csr.tocsr(copy=True)
    idx = np.atleast_1d(idx).astype(int)
    vals = (np.zeros_like(idx, dtype=float) if values is None
            else (np.full_like(idx, float(values)) if np.isscalar(values)
                  else np.asarray(values, float)))
    assert vals.shape == idx.shape

    # zero rows + set diagonals + set rhs
    for k, val in zip(idx, vals):
        rs, re = A.indptr[k], A.indptr[k+1]
        A.data[rs:re] = 0.0
        cols = A.indices[rs:re]
        hit = np.where(cols == k)[0]
        if hit.size:
            A.data[rs + hit[0]] = 1.0
        else:
            A += sparse.csr_matrix(([1.0], ([k], [k])), shape=A.shape)
        b[k] = val

    if keep_spd:
        Ac = A.tocsc()
        for k in idx:
            cs, ce = Ac.indptr[k], Ac.indptr[k+1]
            rows = Ac.indices[cs:ce]
            mask = rows != k
            Ac.data[cs:ce][mask] = 0.0
        A = Ac.tocsr()

    return A, b

def apply_atmospheric_on_polyline_eta(A_csr, b, grid, eta, patm=0.0, masks=None):
    """
    Anchor p = patm along the free surface polyline z = eta(r).
    For each radial column i, we anchor the cell center just BELOW eta[i].
    If masks.solid_cell is provided, we step downward until we hit a fluid cell.

    Parameters
    ----------
    A_csr : scipy.sparse.csr_matrix
        Pressure Poisson matrix (CSR).
    b : ndarray (Nr*Nz,)
        RHS vector (Fortran ordering).
    grid : object with r_c, z_c, r_edges, z_edges, Nr, Nz
    eta : ndarray (Nr,)
        Free-surface height at r_c.
    patm : float
        Atmospheric pressure value to impose.
    masks : dict or None
        If provided, use masks['solid_cell'] (Nr,Nz) to avoid solid cells.

    Returns
    -------
    A_new, b_new : CSR matrix, ndarray
        Matrix/vector with Dirichlet anchors applied.
    """
    Nr, Nz = grid.Nr, grid.Nz
    z_edges = grid.z_edges

    solid = masks.get("solid_cell", None) if masks is not None else None

    # for each column i, find j_top below eta[i]
    j_top = np.searchsorted(z_edges, eta, side='right') - 1
    j_top = np.clip(j_top, 0, Nz-1)

    # if that cell is solid, step downward until fluid (or give up)
    if solid is not None:
        for i in range(Nr):
            j = j_top[i]
            while j >= 0 and solid[i, j]:
                j -= 1
            j_top[i] = max(j, 0)

    # build list of DOF indices to anchor
    idx = np.array([_p_dof_index(i, j_top[i], Nr) for i in range(Nr)], dtype=int)
    vals = np.full_like(idx, float(patm), dtype=float)

    A_new, b_new = anchor_pressure_exact_many(A_csr, b, idx, values=vals, keep_spd=True)
    return A_new, b_new
    

def simple_solve_swirl(Nr=40, Nz=40, R=0.1, H=5.0, Zmin=0.0, Zmax=0.3,
                       rho=1000.0, mu=0.0, g=9.81, eta0=None,
                       dt=0.01, max_iter=500, alpha_p=0.3,
                       tol_div=1e-8, verbose=True,
                       bc: Dict[str, Dict[str, Tuple[str, Profile]]] | None = None,
                       u_t_init: Array | None = None,
                       plot_freq = 0, dump_freq = 0, run_label = '', showWall=True,
                       do_wall=True, do_injector=True, do_drain=True,
                       ):
    """SIMPLE loop with u_theta transport. Returns fields and history."""
    if bc is None: bc = {}
    grid = build_grid(Nr, Nz, R, Zmin, Zmax)
    dr = grid.dr.mean(); dz = grid.dz.mean()
    nu = mu / rho if mu>0 else 0.0

    masks = build_slanted_wall_masks(grid, R, z_min=0.0, z_max=4.0,
                                     do_wall = do_wall)

    # free-surface needs an initial guess
    if eta0 is None:
        eta = np.full(Nr, H) # guess flat and go for it
    else:
        eta = eta0.copy()
        
    # Fields
    p = np.zeros((Nr, Nz))
    u_r = np.zeros((Nr+1, Nz))
    u_z = np.zeros((Nr,   Nz+1))
    u_t = np.zeros((Nr, Nz)) if u_t_init is None else u_t_init.copy()

    # Apply initial u_t BCs
    apply_ut_boundary_segmented(u_t, grid, bc)

    # Poisson operator (constant ρ)
    Ap = assemble_poisson_matrix(grid, rho)

    history = []
    for it in range(1, max_iter+1):

        # Predictor for meridional velocities
        dpdr, dpdz = grad_p_on_faces(p, grid)
        u_r_star = u_r - (dt/rho)*dpdr
        u_z_star = u_z - (dt/rho)*dpdz - dt*g

        # --- NEW: add centrifugal acceleration from swirl on radial faces ---
        # face radii r_f = r_edges[1:-1] for interior faces
        r_f = grid.r_edges                   # shape (Nr+1,)
        # interpolate u_theta from centers to the two cells adjacent to each radial face
        # then average to the face; result shape (Nr-1, Nz)
        u_t_left  = u_t[:-1, :]                    # (Nr-1, Nz)
        u_t_right = u_t[1:,  :]                    # (Nr-1, Nz)
        u_t_face  = 0.5*(u_t_left + u_t_right)     # (Nr-1, Nz) centered on interior faces
        # a_c = u_theta^2 / r at the face
        a_c = np.zeros_like(u_r_star)
        a_c[1:-1,:] = (u_t_face**2) / r_f[1:-1, None]     # broadcast r_f along z
        # apply mask to hide centrifugal force in the walled off portion
        a_c *= masks['face_open_r'].astype(float)
        # apply to interior radial faces; boundaries remain whatever BCs enforce
        u_r_star += dt * a_c
        
        if nu > 0.0:
            u_r_star += dt * nu * laplacian_faces(u_r, dr, dz, orient='r')
            u_z_star += dt * nu * laplacian_faces(u_z, dr, dz, orient='z')

        # Apply boundary normal velocities
        apply_boundary_normal_segmented(u_r_star, u_z_star, grid, bc)

        # set up BC's from injector and drain
        if do_injector:
            Q_in = integrate_wall_inflow_rR(bc, grid)
        if do_drain and do_injector:
            u_bottom = make_bottom_drain_profile_auto(bc, grid, Q_in)
        else:
            u_bottom = None
        if u_bottom is not None:
            u_z_star[:, 0] = u_bottom     # impose predictor normal velocity at bottom


        # BACK TO THE PAST
        # # NEW: enforce internal slanted wall (no penetration + u_theta no-slip)
        # apply_internal_wall_bc(u_r_star, u_z_star, u_t, grid, masks)

        # BACK TO THE PAST
        # # build pressure-correction Poisson RHS with segmented Neumann terms
        # b = poisson_rhs_with_bc_segmented_masked(grid, u_r_star, u_z_star, dt, rho, bc,
        #                                          masks, u_bottom=u_bottom,
        #                                          rhs_builder_no_mask=poisson_rhs_with_bc_segmented)

        div_u = divergence_from_face_fluxes(u_r_star, u_z_star, grid) # BACK TO THE PAST
        b = -div_u.ravel(order='F')/dt

        # BACK TO THE PAST
        # get liquid-air interface by solving for eta(r)
        # A_p, b_p = apply_atmospheric_on_polyline_eta(Ap.copy(), b.copy(), grid, eta, patm=1e5, masks=masks)
        # solve Poisson eqn for P'
        # p_prime = spsolve(A_p, b_p).reshape((Nr, Nz), order='F')

        # BACK TO THE PAST
        # Solve Poisson
        Ap_anch, b_anch = anchor_pressure(Ap.copy(), b.copy(), k=0)
        p_prime = spsolve(Ap_anch, b_anch).reshape((Nr, Nz), order='F')

        # Update pressure and correct velocities with p_prime
        oldP = p.copy()
        p += alpha_p * p_prime
        dpdr_p, dpdz_p = grad_p_on_faces(p_prime, grid)
        u_r = u_r_star - (dt/rho) * dpdr_p
        u_z = u_z_star - (dt/rho) * dpdz_p

        # Re-enforce boundary normals
        apply_boundary_normal_segmented(u_r, u_z, grid, bc)

        # Update u_theta explicitly with current (u_r,u_z)
        # u_t_old = u_t.copy()
        u_t = _update_utheta(u_t, u_r, u_z, grid, nu, dt, bc)
        # print("max ut change = ", np.max(np.abs(u_t - u_t_old)))

        # Diagnostics
        # BACK TO THE PAST
        # div_u = divergence_from_face_fluxes_masked(u_r, u_z, grid, masks)
        div_u = divergence_from_face_fluxes(u_r, u_z, grid) # BACK TO THE PAST
        max_div = np.max(np.abs(div_u))
        pc = p_prime.copy()
        pc -= pc.mean()
        res_p = np.max(np.abs(pc))
        # residual for dp/dr, dp/dz correctness
        dpdr_c = 0.5*(dpdr[:-1,:] + dpdr[1:,:])
        dpdz_c = 0.5*(dpdz[:, :-1] + dpdz[:, 1:])
        dpdz_c[:,0] = dpdz_c[:,1]
        dpdz_c[:,-1] = dpdz_c[:,-2]
        Rr = dpdr_c - rho*(u_t**2)/np.maximum(grid.r_c[:,None], 1e-12)
        Rr[Nr-1,:] = 0.0 # boundary values don't mean much here
        Rz = dpdz_c + rho*g
        # print(it, " dpdr_c[-4:-1,0] = ", dpdr_c[-4:,0])
        # print(it, " dpdr[-4:-1,0] = ", dpdr[-4:,0])
        # print(it, " rho*(u_t**2)/r[-4:-1,0] = ", rho*(u_t[-4:,0]**2)/grid.r_c[-4:])
        res_mom = max(np.abs(Rr).max(), np.abs(Rz).max())
        res_mom_r = (np.abs(Rr)).max()
        res_mom_z = (np.abs(Rz)).max()

        # Track u_t change (Linf)
        # hist_ut = np.max(np.abs(_update_utheta(u_t, u_r, u_z, grid, nu, 0.0, bc) - u_t))  # zero dt -> just BC reapply
        # history.append((it, max_div, res_p, hist_ut))

        # update the surface
        # IN THIS BACK TO THE PAST VERSION THIS ETA HAS NO EFFECT ON
        # ANYTHING except plots
        eta, eta_residual = kinematic_eta_update(eta, u_r, u_z, grid, dt, lam=0.4)

        if verbose and (it % 1000 == 0 or it == 1):
            # print(f"Iter {it:4d}: max(div u)={max_div:.3e} res_p={res_p:.3e} max|ur|={np.max(np.abs(u_r)):.3e} max|uz|={np.max(np.abs(u_z)):.3e}")
            # print("eta_residual = ", eta_residual)
            print(f"Iter {it:4d}: max(div u)={max_div:.3e} res_p={res_p:.3e} res_mom_r={res_mom_r:.3e} res_mom_z={res_mom_z:.3e} eta_res={np.sum(eta_residual):.3e}")

        if (it % plot_freq == 0 or it == 1):

            # # # diagnostic plot of dpdz = - rho g
            # plot_pressure_contours(Rr, grid,
            #                        title='dpdr - ut**2/r contours %6d iters' % (it),
            #                        unit='Pa/m', scale=1, n_contours=10,
            #                        filename=None, show=True,
            #                        eta=eta, showWall=showWall)
            
            # -------------------
            # Plot: pressure colormap + u_theta contours
            # -------------------
            # BACK TO THE PAST
            # adjust to keep mass conservation
            V_target = np.pi * (grid.R**2) * H          # your known fill volume
            c = choose_pressure_offset_for_volume(p, grid, V_target, masks, patm=0.0)
            p += c # now the P=0 contour will contain our mass
            plot_pressure_contours(p, grid,
                                   title='injected fluid: pressure contours %6d iters' % (it),
                                   unit='bar', scale=1e5, n_contours=10,
                                   filename="%s_Pcontours_%06d.png"%(run_label,it), show=False,
                                   eta=eta, showWall=showWall)
            
            plot_streamlines(u_r, u_z, grid, density=1.5,
                             title='Streamlines (u_r, u_z) %6d iters' % (it),
                             filename="%s_streamlines_%06d.png"%(run_label,it), show=False,
                             eta=eta, showWall=showWall,)
            
            plot_pressure_with_swirl(p, u_t, grid,
                                     title='u_theta(r,z) %6d iters' % (it),
                                     unit='m/s', scale=1.0, show_contours=False,
                                     filename="%s_u_theta_%06d.png"%(run_label,it),
                                     eta=eta, show=False, showWall=showWall)
            
        # if max_div < tol_div and res_p < 1e-8:
        if max_div < tol_div and res_p < 1 and res_mom < 1e-6 * rho * g:
            if verbose:
                print(f"Converged (continuity & pressure) at iter {it}: max(div u)={max_div:.3e}")
            break

    # print(f"Iter {it:4d}: max(div u)={max_div:.3e} max|p'-oldP|={res_p:.3e} max|ur|={np.max(np.abs(u_r)):.3e} max|uz|={np.max(np.abs(u_z)):.3e} eta_res={eta_residual:.3e}")
    print(f"Iter {it:4d}: max(div u)={max_div:.3e} res_p={res_p:.3e} res_mom_r={res_mom_r:.3e} res_mom_z={res_mom_z:.3e} eta_res={np.sum(eta_residual):.3e}")


    return {'p': p, 'u_r': u_r, 'u_z': u_z, 'u_t': u_t, 'grid': grid,
            'history': history, 'eta': eta}

def NEW_simple_solve_swirl(Nr=40, Nz=40, R=0.1, H=5.0, Zmin=0.0, Zmax=0.3,
                       rho=1000.0, mu=0.0, g=9.81, eta0=None,
                       dt=0.01, max_iter=500, alpha_p=0.3,
                       tol_div=1e-8, verbose=True,
                       bc: Dict[str, Dict[str, Tuple[str, Profile]]] | None = None,
                       u_t_init: Array | None = None,
                       plot_freq = 0, dump_freq = 0, run_label = '', showWall=True,
                       ):
    """SIMPLE loop with u_theta transport. Returns fields and history."""
    if bc is None: bc = {}
    grid = build_grid(Nr, Nz, R, Zmin, Zmax)
    dr = grid.dr.mean(); dz = grid.dz.mean()
    nu = mu / rho if mu>0 else 0.0
    masks = build_slanted_wall_masks(grid, R, z_min=0.0, z_max=4.0)

    # free-surface needs an initial guess
    if eta0 is None:
        eta = np.full(Nr, H) # guess flat and go for it
    else:
        eta = eta0.copy()
        
    # Fields
    p = np.zeros((Nr, Nz))
    u_r = np.zeros((Nr+1, Nz))
    u_z = np.zeros((Nr,   Nz+1))
    u_t = np.zeros((Nr, Nz)) if u_t_init is None else u_t_init.copy()

    # Apply initial u_t BCs
    apply_ut_boundary_segmented(u_t, grid, bc)

    # Poisson operator (constant ρ)
    Ap = assemble_poisson_matrix(grid, rho)

    history = []
    for it in range(1, max_iter+1):

        # Predictor for meridional velocities
        dpdr, dpdz = grad_p_on_faces(p, grid)
        u_r_star = u_r - (dt/rho)*dpdr
        u_z_star = u_z - (dt/rho)*dpdz - dt*g

        # --- NEW: add centrifugal acceleration from swirl on radial faces ---
        # face radii r_f = r_edges[1:-1] for interior faces
        r_f = grid.r_edges                   # shape (Nr+1,)
        # interpolate u_theta from centers to the two cells adjacent to each radial face
        # then average to the face; result shape (Nr-1, Nz)
        u_t_left  = u_t[:-1, :]                    # (Nr-1, Nz)
        u_t_right = u_t[1:,  :]                    # (Nr-1, Nz)
        u_t_face  = 0.5*(u_t_left + u_t_right)     # (Nr-1, Nz) centered on interior faces
        # a_c = u_theta^2 / r at the face
        a_c = np.zeros_like(u_r_star)
        a_c[1:-1,:] = (u_t_face**2) / r_f[1:-1, None]     # broadcast r_f along z
        # apply mask to hide centrifugal force in the walled off portion
        a_c *= masks['face_open_r'].astype(float)
        # apply to interior radial faces; boundaries remain whatever BCs enforce
        u_r_star += dt * a_c
        
        if nu > 0.0:
            u_r_star += dt * nu * laplacian_faces(u_r, dr, dz, orient='r')
            u_z_star += dt * nu * laplacian_faces(u_z, dr, dz, orient='z')

        # Apply boundary normal velocities
        apply_boundary_normal_segmented(u_r_star, u_z_star, grid, bc)

        # set up BC's from injector and drain
        Q_in = integrate_wall_inflow_rR(bc, grid)
        u_bottom = make_bottom_drain_profile_auto(bc, grid, Q_in)
        if u_bottom is not None:
            u_z_star[:, 0] = u_bottom     # impose predictor normal velocity at bottom


        # NEW: enforce internal slanted wall (no penetration + u_theta no-slip)
        apply_internal_wall_bc(u_r_star, u_z_star, u_t, grid, masks)
        
        # build pressure-correction Poisson RHS with segmented Neumann terms
        b = poisson_rhs_with_bc_segmented_masked(grid, u_r_star, u_z_star, dt, rho, bc,
                                                 masks, u_bottom=u_bottom,
                                                 rhs_builder_no_mask=poisson_rhs_with_bc_segmented)

        # get liquid-air interface by solving for eta(r)
        A_p, b_p = apply_atmospheric_on_polyline_eta(Ap.copy(), b.copy(), grid, eta, patm=1e5, masks=masks)
        # solve Poisson eqn for P'
        p_prime = spsolve(A_p, b_p).reshape((Nr, Nz), order='F')

        # # Solve Poisson
        # Ap_anch, b_anch = anchor_pressure(Ap.copy(), b.copy(), k=0)
        # p_prime = spsolve(Ap_anch, b_anch).reshape((Nr, Nz), order='F')

        # Update pressure and correct velocities with p_prime
        oldP = p.copy()
        p += alpha_p * p_prime
        dpdr_p, dpdz_p = grad_p_on_faces(p_prime, grid)
        u_r = u_r_star - (dt/rho) * dpdr_p
        u_z = u_z_star - (dt/rho) * dpdz_p

        # Re-enforce boundary normals
        apply_boundary_normal_segmented(u_r, u_z, grid, bc)

        # Update u_theta explicitly with current (u_r,u_z)
        u_t = _update_utheta(u_t, u_r, u_z, grid, nu, dt, bc)

        # Diagnostics
        div_u = divergence_from_face_fluxes_masked(u_r, u_z, grid, masks)
        max_div = np.max(np.abs(div_u))
        res_p = np.max(np.abs(p-oldP))
        # Track u_t change (Linf)
        hist_ut = np.max(np.abs(_update_utheta(u_t, u_r, u_z, grid, nu, 0.0, bc) - u_t))  # zero dt -> just BC reapply
        history.append((it, max_div, res_p, hist_ut))

        # update the surface
        eta, eta_residual = kinematic_eta_update(eta, u_r, u_z, grid, dt, lam=0.4)

        if verbose and (it % 1000 == 0 or it == 1):
            # print(f"Iter {it:4d}: max(div u)={max_div:.3e} max|p'-oldP|={res_p:.3e} max|ur|={np.max(np.abs(u_r)):.3e} max|uz|={np.max(np.abs(u_z)):.3e}")
            print("eta_residual = ", eta_residual)
            print(f"Iter {it:4d}: max(div u)={max_div:.3e} max|p'-oldP|={res_p:.3e} eta_res={np.sum(eta_residual):.3e}")

        if (it % plot_freq == 0 or it == 1):
            # -------------------
            # Plot: pressure colormap + u_theta contours
            # -------------------
            # adjust to keep mass conservation
            # V_target = np.pi * (grid.R**2) * H          # your known fill volume
            # c = choose_pressure_offset_for_volume(p, grid, V_target,
            #                                       masks, patm=0.0)
            # p += c # now the P=0 contour will contain our mass
            plot_pressure_contours(p, grid,
                                   title='injected fluid: pressure contours %6d iters' % (it),
                                   unit='bar', scale=1e5, n_contours=10,
                                   filename="%s_Pcontours_%06d.png"%(run_label,it), show=False,
                                   eta=eta, showWall=showWall)
            
            plot_streamlines(u_r, u_z, grid, density=1.5,
                             title='Streamlines (u_r, u_z) %6d iters' % (it),
                             filename="%s_streamlines_%06d.png"%(run_label,it), show=False,
                             eta=eta, showWall=showWall,)
            
            plot_pressure_with_swirl(p, u_t, grid,
                                     title='u_theta(r,z) %6d iters' % (it),
                                     unit='m/s', scale=1.0, show_contours=False,
                                     filename="%s_u_theta_%06d.png"%(run_label,it),
                                     eta=eta, show=False, showWall=showWall)
            
        # if max_div < tol_div and res_p < 1e-8:
        if max_div < tol_div and res_p < 1:
            if verbose:
                print(f"Converged (continuity & pressure) at iter {it}: max(div u)={max_div:.3e}")
            break

    # print(f"Iter {it:4d}: max(div u)={max_div:.3e} max|p'-oldP|={res_p:.3e} max|ur|={np.max(np.abs(u_r)):.3e} max|uz|={np.max(np.abs(u_z)):.3e} eta_res={eta_residual:.3e}")
    print(f"Iter {it:4d}: max(div u)={max_div:.3e} max|p'-oldP|={res_p:.3e} eta_res={np.sum(eta_residual):.3e}")

    return {'p': p, 'u_r': u_r, 'u_z': u_z, 'u_t': u_t, 'grid': grid,
            'history': history, 'eta': eta}

import matplotlib.pyplot as plt
    
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

    
