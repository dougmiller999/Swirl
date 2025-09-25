import numpy as np

def _minmod(a, b):
    return np.where(a*b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))

def _slope_minmod_nonuniform(y, x):
    """
    Minmod slope on a 1D non-uniform grid.
    y, x: 1D arrays length N.
    Returns dy/dx at the N points (limited).
    """
    N = len(y)
    s = np.zeros_like(y, dtype=float)
    # one-sided at ends
    s[0]  = (y[1]-y[0]) / (x[1]-x[0])
    s[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
    # interior using minmod of left/right one-sided slopes
    dl = (y[1:-1]-y[:-2]) / (x[1:-1]-x[:-2])
    dr = (y[2:]-y[1:-1]) / (x[2:]-x[1:-1])
    s[1:-1] = _minmod(dl, dr)
    return s

def kinematic_eta_update(eta, u_r, u_z, grid, dt, lam=0.4):
    """
    Advance the free surface eta by the kinematic boundary condition:
        d eta/dt + u_r * d eta/dr = u_z   (evaluated at z = eta(r))
    using a simple explicit step with under-relaxation 'lam'.

    Parameters
    ----------
    eta : (Nr,) current free-surface height on r_c
    u_r : (Nr+1, Nz) radial-face velocities
    u_z : (Nr,   Nz+1) axial-face velocities
    grid: has r_c, z_edges
    dt  : time step
    lam : under-relaxation for robustness (0<lam<=1). Try 0.2..0.5

    Returns
    -------
    eta_new : (Nr,)
    residual : (Nr,) kinematic residual R = u_z_surf - u_r_surf * d_eta/dr
    """
    r_c, z_edges = grid.r_c, grid.z_edges
    Nr, Nzp1 = u_z.shape
    Nz = Nzp1 - 1

    # 0) compute dt
    eta_dt = min(np.min(grid.dr_e),np.min(grid.dz_n))/ \
             max(np.max(np.abs(u_r)), np.max(np.abs(u_z)))
    # print("dt = ", dt, " eta_dt = ", eta_dt)

    # 1) slopes d eta / dr (limited)
    deta_dr = _slope_minmod_nonuniform(eta, r_c)

    # 2) indices and weights in z for interpolation at surface height
    jf = np.searchsorted(z_edges, eta, side='right') - 1  # face below eta
    jf = np.clip(jf, 0, Nz-2)
    print(" jf = ", jf)
    dzf = z_edges[jf+1] - z_edges[jf]
    wz = (eta - z_edges[jf]) / np.where(dzf != 0.0, dzf, 1.0)
    wz = np.clip(wz, 0.0, 1.0)

    # 3) u_z at (r_i, eta_i): interpolate between axial faces jf and jf+1
    # print(" wz = ", wz)
    uz_low  = u_z[np.arange(Nr), jf]       # (Nr,)
    uz_high = u_z[np.arange(Nr), jf+1]     # (Nr,)
    u_z_surf = (1.0 - wz) * uz_low + wz * uz_high

    # 4) u_r at (r_i, eta_i): first move u_r faces to cell-centers,
    #    then interpolate in z just like for u_z.
    #    cell-centered u_r_c has shape (Nr, Nz): average adjacent faces in r.
    u_r_c = 0.5 * (u_r[:-1, :] + u_r[1:, :])   # (Nr, Nz)

    ur_low  = u_r_c[np.arange(Nr), jf]         # (Nr,)
    ur_high = u_r_c[np.arange(Nr), jf+1]       # (Nr,)
    u_r_surf = (1.0 - wz) * ur_low + wz * ur_high

    # 5) kinematic residual and update
    residual = u_z_surf - u_r_surf * deta_dr
    # print(" u_z_surf = ", u_z_surf)
    # print(" u_r_surf = ", u_r_surf)
    # print(" u_r_surf*deta_dr = ", u_r_surf*deta_dr)
    # eta_new = eta + lam * dt * residual
    eta_new = eta + lam * eta_dt * residual

    # Clamp inside domain bounds for safety
    eta_new = np.clip(eta_new, z_edges[0], z_edges[-1])

    return eta_new, residual

import numpy as np

def default_eta0(grid, R, H_mean, V_target, Omega=0.0, g=9.81):
    """
    Build an initial guess for the free surface eta(r).

    Parameters
    ----------
    grid : object
        Must provide r_c, r_edges (axisymmetric grid).
    R : float
        Outer radius of bucket [m].
    H_mean : float
        Mean fill height [m] (V_target / (pi R^2)).
    V_target : float
        Known total liquid volume [m^3].
    Omega : float, optional
        Spin rate [rad/s]. If 0, returns a flat surface at H_mean.
    g : float, optional
        Gravitational acceleration [m/s^2].

    Returns
    -------
    eta0 : ndarray, shape (Nr,)
        Initial surface profile at radial centers.
    """

    r_c = grid.r_c
    Nr  = grid.Nr
    dr  = grid.r_edges[1:] - grid.r_edges[:-1]

    if Omega == 0.0:
        # flat fill
        return np.full(Nr, H_mean)

    # raw paraboloid profile: eta(r) = h0 + (Ω² r²)/(2g)
    h0 = 0.0  # temporary; we'll adjust offset below
    eta = h0 + 0.5 * (Omega**2 / g) * r_c**2

    # adjust offset so volume matches V_target
    # Volume under surface: integrate annuli 2π r * dr * eta(r)
    V_eta = 2*np.pi * np.sum(r_c * eta * dr)
    scale = V_target / V_eta
    eta *= scale

    return eta
