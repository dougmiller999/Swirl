
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
import sys
sys.path.append('..')

import numpy as np
from simple_solver_swirl import simple_solve_swirl
from graphics import (
    plot_pressure_with_swirl, plot_streamlines,
    plot_pressure_contours,
    choose_pressure_offset_for_volume
)
from axisym_poisson_projection import build_grid

import cProfile

do_graphics = True
show = False # show plots during the run in addition to saving them
show = True # show plots during the run in addition to saving them
plot_freq = 200

# -------------------
# Problem parameters
# -------------------
g    = 9.81  # m/s^2
rho = 1000.0            # kg/m^3
mu  = 1.0e-3            # Pa·s
R   = 3.0               # m
H   = 5.0               # m (mean height)
Zmin = 0.0

run = "slow"

if run == "fast":
    # this piles up liquid between 2.23 and R in a sharp near-triangle
    run_label = 'fast_bucket'
    rpm = 100.0
    Omega = rpm * 2.0*np.pi / 60.0   # rad/s
    Zmax = 25.0 # m, z-range allowed
else:
    # changing to one where there is no air cone
    run_label = 'slow_bucket'
    Omega = 4.5  # rad/s
    Zmax = 10.0 # m, z-range allowed

# -------------------
# Numerical settings
# -------------------
Nr, Nz = 120, 200       # grid resolution (tune as desired)
Nr, Nz = 2*30, 2*50       # grid resolution (tune as desired)
Nr, Nz = 30, 50       # grid resolution (tune as desired)
dt = 0.0004              # s (keep CFL = u*dt/dr <= ~0.5; here u ~ ΩR ≈ 31.4 m/s → dr ≈ R/Nr)
dt = 0.001 # cheating
max_iter = 1000000          # SIMPLE iterations
max_iter = 10         # SIMPLE iterations
max_iter = 200        # SIMPLE iterations
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

ut_init = np.ones((Nr,Nz), dtype=float)
grid = build_grid(Nr, Nz, R, Zmin, Zmax)
for j in range(Nz):
    ut_init[:,j] = Omega * grid.r_c

# -------------------
# Solve
# -------------------
# profiler = cProfile.Profile()
# profiler.enable()

out = simple_solve_swirl(Nr=Nr, Nz=Nz, R=R, H=H, Zmin=0.0, Zmax=Zmax,
                         rho=rho, mu=mu, g=g, eta0=None,
                         dt=dt, max_iter=max_iter, alpha_p=alpha_p, tol_div=tol_div, verbose=True,
                         bc=bc,
                         do_wall= False, # no internal walls in basic bucket
                         do_injector = False, # no injectors for basic bucket
                         do_drain = False,    # no drain for basic bucket
                         # u_t_init=None,
                         u_t_init=ut_init,    # approx solution to start
                         plot_freq=plot_freq, dump_freq=0, run_label =
                         run_label, showWall = False)

# profiler.disable()
# profiler.print_stats(sort='cumulative')

p, u_t, u_r, u_z, grid, eta = out['p'], out['u_t'], out['u_r'],out['u_z'], out['grid'], out['eta']

if do_graphics:
    # BACK TO THE PAST
    # adjust to keep mass conservation
    V_target = np.pi * (grid.R**2) * H          # your known fill volume
    masks = {'solid_cell' : np.zeros((Nr,Nz),dtype=bool)}
    c = choose_pressure_offset_for_volume(p, grid, V_target, masks, patm=0.0)
    p += c # now the P=0 contour will contain our mass

    # actually we know this solution, so let's plot it
    if run == 'fast':
        r0 = 2.233
        r = grid.r_c
        eta= H/(1-(r0/R)**2) + Omega**2*R**2/(2*g) * ((r/R)**2 - 0.5*(1+(r0/R)**2))
        eta = np.clip(eta, 0.0, 25.0)
    elif run == 'slow':
        r0 = 0.0
        r = grid.r_c
        eta= H/(1-(r0/R)**2) + Omega**2*R**2/(2*g) * ((r/R)**2 - 0.5*(1+(r0/R)**2))
        eta = np.clip(eta, 0.0, 25.0)
    # -------------------
    # Plot: pressure colormap
    # -------------------
    plot_pressure_contours(p, grid,
                           title='Rotating bucket: pressure contours',
                           unit='bar', scale=1e5, n_contours=10,
                           filename="%s_Pcontours.png"%(run_label), show=show,
                           eta=eta, showWall=False)

    # -------------------
    # Plot: pressure colormap + u_theta contours
    # -------------------
    plot_pressure_with_swirl(p, u_t, grid,
                             title=f'injected fluid: u_theta (colormap)',
                             show_contours=False, n_contours=18,
                             filename="%s_Ut_contours.png"%(run_label), show=show,
                             eta = eta, showWall = False)

    # -------------------
    # Plot: meridional streamlines
    # -------------------
    plot_streamlines(u_r, u_z, grid, density=1.3, 
                     title=f'Rotating bucket: streamlines',
                     filename="%s_streamlines.png"%(run_label), show=show,
                     eta=eta, showWall=False)


# Print a few basics
print('Omega [rad/s]=', Omega)
print('Max |u_theta| [m/s]=', float(np.max(np.abs(u_t))))
print('Max |u_r|, |u_z| [m/s]=', float(np.max(np.abs(u_r))), float(np.max(np.abs(u_z))))
