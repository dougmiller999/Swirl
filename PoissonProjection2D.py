# Re-run the full code cell (the execution state was reset).

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg
import matplotlib.pyplot as plt

def build_grid(Nr, Nz, R, Zmin, Zmax):
    r_edges = np.linspace(0.0, R, Nr+1)          # size Nr+1
    z_edges = np.linspace(Zmin, Zmax, Nz+1)      # size Nz+1
    r_c = 0.5*(r_edges[:-1] + r_edges[1:])       # size Nr
    z_c = 0.5*(z_edges[:-1] + z_edges[1:])       # size Nz
    dr = np.diff(r_edges)                         # size Nr
    dz = np.diff(z_edges)                         # size Nz
    return r_edges, z_edges, r_c, z_c, dr, dz

def geom_areas_volumes(r_edges, z_edges, r_c, z_c, dr, dz):
    Nr, Nz = len(r_c), len(z_c)
    rP = r_c[:, None] * np.ones((Nr, Nz))
    dr_i = dr[:, None] * np.ones((Nr, Nz))
    dz_j = np.ones((Nr, Nz)) * dz[None, :]
    V = (2*np.pi) * rP * dr_i * dz_j
    A_e = (2*np.pi) * r_edges[1:, None] * dz_j
    A_w = (2*np.pi) * r_edges[:-1, None] * dz_j
    A_n = (2*np.pi) * rP * dr_i
    A_s = (2*np.pi) * rP * dr_i
    return A_e, A_w, A_n, A_s, V

def assemble_poisson_matrix(Nr, Nz, r_edges, z_edges, r_c, z_c, dr, dz, rho):
    def lin(i,j): return i + Nr*j
    Nc = Nr*Nz
    Ap = lil_matrix((Nc, Nc), dtype=float)

    dr_e = np.empty(Nr); dr_e[:-1] = r_c[1:] - r_c[:-1]; dr_e[-1] = np.nan
    dr_w = np.empty(Nr); dr_w[1:]  = r_c[1:] - r_c[:-1]; dr_w[0]  = np.nan
    dz_n = np.empty(Nz); dz_n[:-1] = z_c[1:] - z_c[:-1]; dz_n[-1] = np.nan
    dz_s = np.empty(Nz); dz_s[1:]  = z_c[1:] - z_c[:-1]; dz_s[0]  = np.nan

    A_e, A_w, A_n, A_s, V = geom_areas_volumes(r_edges, z_edges, r_c, z_c, dr, dz)

    for j in range(Nz):
        for i in range(Nr):
            row = lin(i,j)
            if i < Nr-1:
                ae = (A_e[i,j]) / (rho * dr_e[i])
                Ap[row, lin(i+1,j)] += -ae
                Ap[row, row]        +=  ae
            if i > 0:
                aw = (A_w[i,j]) / (rho * dr_w[i])
                Ap[row, lin(i-1,j)] += -aw
                Ap[row, row]        +=  aw
            if j < Nz-1:
                an = (A_n[i,j]) / (rho * dz_n[j])
                Ap[row, lin(i,j+1)] += -an
                Ap[row, row]        +=  an
            if j > 0:
                as_ = (A_s[i,j]) / (rho * dz_s[j])
                Ap[row, lin(i,j-1)] += -as_
                Ap[row, row]        +=  as_
    return Ap.tocsr()

def divergence_from_face_fluxes(u_r, u_z, A_e, A_w, A_n, A_s, V):
    Nr, Nz = A_e.shape
    div = np.zeros((Nr, Nz))
    for j in range(Nz):
        for i in range(Nr):
            Phi_e = u_r[i+1, j] * A_e[i, j]
            Phi_w = u_r[i,   j] * A_w[i, j]
            Phi_n = u_z[i, j+1] * A_n[i, j]
            Phi_s = u_z[i, j  ] * A_s[i, j]
            div[i,j] = (Phi_e - Phi_w + Phi_n - Phi_s) / V[i,j]
    return div

def correct_face_velocities(u_r_star, u_z_star, p, r_c, z_c, rho, dt):
    Nr, Nz = p.shape
    u_r_new = u_r_star.copy()
    for j in range(Nz):
        for i_face in range(1, Nr):
            iW = i_face - 1; iE = i_face
            dr_face = r_c[iE] - r_c[iW]
            dpdr = (p[iE, j] - p[iW, j]) / dr_face
            u_r_new[i_face, j] = u_r_star[i_face, j] - (dt / rho) * dpdr
    u_z_new = u_z_star.copy()
    for i in range(Nr):
        for j_face in range(1, Nz):
            jS = j_face - 1; jN = j_face
            dz_face = z_c[jN] - z_c[jS]
            dpdz = (p[i, jN] - p[i, jS]) / dz_face
            u_z_new[i, j_face] = u_z_star[i, j_face] - (dt / rho) * dpdz
    return u_r_new, u_z_new

# Parameters and grid
Nr, Nz = 20, 30
R, Zmin, Zmax = 0.1, 0.0, 0.3
rho, g = 1000.0, 9.81
dt = 0.05

r_edges, z_edges, r_c, z_c, dr, dz = build_grid(Nr, Nz, R, Zmin, Zmax)
A_e, A_w, A_n, A_s, V = geom_areas_volumes(r_edges, z_edges, r_c, z_c, dr, dz)

# Predictor
u_r_star = np.zeros((Nr+1, Nz))
u_z_star = np.zeros((Nr,   Nz+1))
u_z_star[:, 1:-1] = -g * dt  # interior faces only

# Poisson assembly
Ap = assemble_poisson_matrix(Nr, Nz, r_edges, z_edges, r_c, z_c, dr, dz, rho)

# RHS
div_u_star = divergence_from_face_fluxes(u_r_star, u_z_star, A_e, A_w, A_n, A_s, V)
b = (div_u_star.flatten(order='F')) / dt

# Anchor pressure
Ap = Ap.tolil()
row_ref = 0
Ap.rows[row_ref] = [row_ref]
Ap.data[row_ref] = [1.0]
Ap = Ap.tocsr()
b[row_ref] = 0.0

# Solve
p = spsolve(Ap, b).reshape((Nr, Nz), order='F')

# Correct and enforce impermeable walls
u_r_new, u_z_new = correct_face_velocities(u_r_star, u_z_star, p, r_c, z_c, rho, dt)
u_r_new[0,  :] = 0.0
u_r_new[-1, :] = 0.0
u_z_new[:,  0] = 0.0
u_z_new[:, -1] = 0.0

# Divergence check
div_u_new = divergence_from_face_fluxes(u_r_new, u_z_new, A_e, A_w, A_n, A_s, V)
max_div = np.max(np.abs(div_u_new))

# p(z) line at mid-radius
i_mid = Nr//2
z = z_c
p_col = p[i_mid, :]
Afit = np.vstack([np.ones_like(z), z]).T
coeff, *_ = np.linalg.lstsq(Afit, p_col, rcond=None)
a_fit, b_fit = coeff

plt.figure(figsize=(7,5))
plt.plot(z, p_col/1e5, 'o', label='Numerical p(z) (mid-radius)')
plt.plot(z, (a_fit + b_fit*z)/1e5, '-', label=f'Linear fit slope = {b_fit/1e5:.3f} bar/m')
plt.plot(z, (a_fit - rho*g*z)/1e5, '--', label=f'Expected slope = {-rho*g/1e5:.3f} bar/m')
plt.xlabel('z (m)')
plt.ylabel('Pressure (bar)')
plt.title('Hydrostatic check after projection (staggered grid)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Max cell divergence after projection: {max_div:.3e} 1/s")
print(f"Max |u_r| after projection: {np.max(np.abs(u_r_new)):.3e} m/s")
print(f"Max |u_z| after projection: {np.max(np.abs(u_z_new)):.3e} m/s")
print(f"Fitted dp/dz = {b_fit:.6e} Pa/m   vs expected = {-rho*g:.6e} Pa/m")
