"""
flow_down_funnel_setup.py

Sets up an injected fluid in axisymmetric (r–z) coordinates and solves with SIMPLE + swirl:
- No-slip walls (side & bottom), open top (no normal flow; zero-grad swirl)
- Density rho = 512 kg/m^3 # liquid lithium at melting point 180 C
- Viscosity mu = 0.0005 Pa·s # 5 milliPoise, at melting point
- Radius R = 3 m
- Mean fill height H = 5 m  (≈141.4 m^3 volume)
- injection velocity is small radial inflow, no theta at all

Requires:
  - simple_solver_swirl.py in the same folder (provided earlier)
  - matplotlib, numpy, scipy

Run:
  python rotating_bucket_setup.py
"""
import numpy as np
from simple_solver_swirl import simple_solve_swirl
from graphics import plot_pressure_with_swirl, plot_streamlines, plot_pressure_contours
from internal_wall_mask_patch import build_slanted_wall_masks
import cProfile

profile = False
graphics = True
show = False

# -------------------
# Problem parameters
# -------------------
g    = 9.81  # m/s^2
rho = 512.0            # kg/m^3
mu  = 0.5e-3            # Pa·s
R   = 3.0               # m
H   = 5.0               # m (mean height)

z0 = 4.0 # lower point in the R wall where fluid is injected
z1 = 6.0 # upper point in the R wall where fluid is injected
ur_inject = -0.4 # m/s

run_label = "flow"

# just plop in fluid with no swirl at all
ut_inject = 0.0
Zmax = 10.0 # m, rigid ceiling for the problem in z

# -------------------
# Numerical settings
# -------------------
Nr, Nz = 120, 200       # grid resolution (tune as desired)
Nr, Nz = 30, 50       # grid resolution (tune as desired)
dt = 0.0004              # s (keep CFL = u*dt/dr <= ~0.5; here u ~ ΩR ≈ 31.4 m/s → dr ≈ R/Nr)
dt = 0.001 # cheating
max_iter = 1000000          # SIMPLE iterations
max_iter = 20000          # SIMPLE iterations
max_iter = 50000          # SIMPLE iterations
max_iter = 100          # SIMPLE iterations
plot_freq=20
dump_freq=0
alpha_p = 0.7           # pressure under-relaxation, 0.3 < alpha_p < 0.7
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

if ur_inject > 0.0:
    raise RuntimeError("ur_inject > 0, you are sucking fluid out of"\
    " the problem with your injector")

def swirl_injection(z):
    return ut_inject*np.ones_line(z)

# this drain shape will get cut off by the funnel wall at grid.wall_rmin
def drain_shape(r):
    return np.exp(-(r/(0.4*R))**2)

bc = {
    'r=0':    {'u_r': ('wall', 0.0), 'ut': ('dirichlet', 0.0)},
    'r=R': {
        # normal velocity on r=R (radial faces)
        'u_r': [
            ('segment', (z0, z1, -ur_inject)),   # inlet/outlet profile on z∈[z0,z1]
        ],
        # azimuthal velocity (swirl) on the same wall
        'ut': [
            ('segment_dirichlet', (z0, z1  , ut_inject)),   # u_theta(R,z) on the segment
        ],
    },
    'z=Zmin': {
        # bottom drain: either a fixed profile or an AUTO drain that balances inflow
        'u_z': ('drain_auto', {
            'profile_r': drain_shape,
            'region': (0, R)}),
        # or explicit: ('segment', (r0, r1, profile_r))
    },
    # keep other walls as before...
    'z=Zmax': {'u_z': ('wall', 0.0), 'ut': ('neumann', 0.0)},
}

wall_zmin = 0.0 # funnel ramp starts at (0, R/2)
wall_zmax = 4.0 # funnel ramp ends   at (4, R)

# first guess at the height of the liquid-air interface
wallH = 6.0
wallR = R
drainH = 0.0
drainR = 0.5
eta = np.full(Nr, H) # first guess for debug run

if profile:
    profiler = cProfile.Profile()
    profiler.enable()

# -------------------
# Solve
# -------------------
out = simple_solve_swirl(
    Nr=Nr, Nz=Nz, R=R, H=H, Zmin=0.0, Zmax=Zmax,
    rho=rho, mu=mu, g=g, eta0 = eta,
    dt=dt, max_iter=max_iter, alpha_p=alpha_p, tol_div=tol_div, verbose=True,
    bc=bc,
    plot_freq = plot_freq, dump_freq = dump_freq,
    run_label = run_label
)

if profile:
    profiler.disable()
    profiler.print_stats(sort='cumulative')

p, u_t, u_r, u_z, grid, eta = out['p'], out['u_t'], out['u_r'], out['u_z'], out['grid'], out['eta']

# # adjust to keep mass conservation
# V_target = np.pi * (grid.R**2) * H          # your known fill volume
# masks = build_slanted_wall_masks(grid, R, z_min=grid.wall_zmin, z_max=grid.wall_zmax)
# c = choose_pressure_offset_for_volume(p, grid, V_target, masks, patm=0.0)
# p += c # now the P=0 contour will contain our mass

if graphics:
    # -------------------
    # Plot: pressure colormap
    # -------------------
    plot_pressure_contours(p, grid,
                           title='injected fluid: pressure contours',
                           unit='bar', scale=1e5, n_contours=10,
                           filename="%s_Pcontours.png"%(run_label), show=show,
                           z_min=wall_zmin, z_max=wall_zmax, eta=eta, showWall=True)

    # -------------------
    # Plot: pressure colormap + u_theta contours
    # -------------------
    plot_pressure_with_swirl(p, u_t, grid,
                             title=f'injected fluid: pressure (colormap) + u_theta contours (100 rpm)',
                             unit='bar', scale=1e5, show_contours=True, n_contours=18,
                             filename="%s_Ut_contours.png"%(run_label), show=show)

    # -------------------
    # Plot: meridional streamlines
    # -------------------
    plot_streamlines(u_r, u_z, grid, density=1.3, title='Meridional streamlines (u_r, u_z)',
                     filename="%s_streamlines.png"%(run_label), show=show)

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
print('Max |u_theta| [m/s]=', float(np.max(np.abs(u_t))))
print('Max |u_r|, |u_z| [m/s]=', float(np.max(np.abs(u_r))), float(np.max(np.abs(u_z))))
