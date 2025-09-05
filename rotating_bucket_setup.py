
"""
rotating_bucket_setup.py

Sets up a rotating bucket in axisymmetric (r–z) coordinates and solves with SIMPLE + swirl:
- No-slip walls (side & bottom), open top (no normal flow; zero-grad swirl)
- Density rho = 1000 kg/m^3
- Viscosity mu = 0.001 Pa·s
- Radius R = 3 m
- Mean fill height H = 5 m  (≈141.4 m^3 volume)
- Bucket rotation 100 rpm (Ω rad/s)

Requires:
  - simple_solver_swirl.py in the same folder (provided earlier)
  - matplotlib, numpy, scipy

Run:
  python rotating_bucket_setup.py
"""
import numpy as np
from simple_solver_swirl import simple_solve_swirl, plot_pressure_with_swirl, plot_streamlines
from simple_solver_swirl import plot_pressure_contours, choose_pressure_offset_for_volume
# -------------------
# Problem parameters
# -------------------
g    = 9.81  # m/s^2
rho = 1000.0            # kg/m^3
mu  = 1.0e-3            # Pa·s
R   = 3.0               # m
H   = 5.0               # m (mean height)

# this piles up liquid between 2.23 and R in a sharp near-triangle
rpm = 100.0
Omega = rpm * 2.0*np.pi / 60.0   # rad/s
Zmax = 25.0 # m, range allowed for the problem in z

# changing to one where there is no air cone
Omega = 4.5  # rad/s
Zmax = 10.0 # m, range allowed for the problem in z

# -------------------
# Numerical settings
# -------------------
Nr, Nz = 120, 200       # grid resolution (tune as desired)
Nr, Nz = 30, 50       # grid resolution (tune as desired)
dt = 0.0004              # s (keep CFL = u*dt/dr <= ~0.5; here u ~ ΩR ≈ 31.4 m/s → dr ≈ R/Nr)
dt = 0.001 # cheating
max_iter = 1000          # SIMPLE iterations
max_iter = 1000000          # SIMPLE iterations
alpha_p = 0.7           # pressure under-relaxation
tol_div = 1e-8

# -------------------
# Boundary conditions
# -------------------
# u_r walls at r=0 (axis) and r=R; u_z walls at z=0 (bottom) and z=H (top).
# Swirl BCs (u_theta):
# - axis r=0: Dirichlet u_t=0 (regularity)
# - wall r=R: Dirichlet u_t = Ω R (no-slip rotating wall)
# - bottom z=0: Dirichlet u_t = Ω r (no-slip rotating bottom)
# - top z=H: zero-gradient (shear-free) for u_t; no penetration for u_z

def ut_bottom_profile(r):
    return Omega * r

bc = {
    'r=0':    {'u_r': ('wall', 0.0), 'ut': ('dirichlet', 0.0)},
    'r=R':    {'u_r': ('wall', 0.0), 'ut': ('dirichlet', lambda z: Omega*R*np.ones_like(z))},
    'z=Zmin': {'u_z': ('wall', 0.0), 'ut': ('dirichlet', ut_bottom_profile)},
    'z=Zmax': {'u_z': ('wall', 0.0), 'ut': ('neumann', 0.0)},
}

# -------------------
# Solve
# -------------------
out = simple_solve_swirl(
    Nr=Nr, Nz=Nz, R=R, H=H, Zmin=0.0, Zmax=Zmax,
    rho=rho, mu=mu, g=g,
    dt=dt, max_iter=max_iter, alpha_p=alpha_p, tol_div=tol_div, verbose=True,
    bc=bc
)
p, u_t, u_r, u_z, grid = out['p'], out['u_t'], out['u_r'], out['u_z'], out['grid']

# adjust to keep mass conservation
V_target = np.pi * (grid.R**2) * H          # your known fill volume
c = choose_pressure_offset_for_volume(p, grid, V_target, patm=0.0)
p += c # now the P=0 contour will contain our mass

# -------------------
# Plot: pressure colormap
# -------------------
plot_pressure_contours(p, grid,
                       title='Rotating bucket: pressure contours',
                       unit='bar', scale=1e5, n_contours=10)

# -------------------
# Plot: pressure colormap + u_theta contours
# -------------------
plot_pressure_with_swirl(p, u_t, grid,
                         title=f'Rotating bucket: pressure (colormap) + u_theta contours (100 rpm)',
                         unit='bar', scale=1e5, show_contours=True, n_contours=18)

# # -------------------
# # Plot: meridional streamlines
# # -------------------
# plot_streamlines(u_r, u_z, grid, density=1.3, title='Meridional streamlines (u_r, u_z)')

# # -------------------
# # Optional: overlay theoretical free surface (solid-body) using plot_utils_rz if present
# # z(r) = z0 + (Ω^2 r^2) / (2g), where z0 is chosen so <z> = H ⇒ z0 = H - Ω^2 R^2/(4g).
# # -------------------
# try:
#     from plot_utils_rz import plot_pressure_rz_with_overlays
#     z0 = H - (Omega**2 * R**2) / (4.0*g)
#     def z_free_of_r(r):
#         return z0 + (Omega**2) * r**2 / (2.0*g)
#     plot_pressure_rz_with_overlays(
#         p, grid.r_edges, grid.z_edges,
#         title='Pressure with theoretical free-surface overlay',
#         unit='bar', scale=1e5,
#         z_free=lambda rc: z_free_of_r(rc),  # callable z(r) evaluated on r-centers
#         r_core=None, show_contours=True, n_contours=16
#     )
# except ImportError:
#     pass

# Print a few basics
print('Omega [rad/s]=', Omega)
print('Max |u_theta| [m/s]=', float(np.max(np.abs(u_t))))
print('Max |u_r|, |u_z| [m/s]=', float(np.max(np.abs(u_r))), float(np.max(np.abs(u_z))))
