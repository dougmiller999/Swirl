
"""
Axisymmetric (r–z) finite-volume pressure–Poisson projection on a staggered grid (with gravity).
...
(See file for full docstring.)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

Array = np.ndarray

@dataclass
class Grid:
    Nr: int; Nz: int
    R: float; Zmin: float; Zmax: float
    r_edges: Array; z_edges: Array
    r_c: Array; z_c: Array
    dr: Array; dz: Array
    A_e: Array; A_w: Array; A_n: Array; A_s: Array
    V: Array

def build_grid(Nr: int, Nz: int, R: float, Zmin: float, Zmax: float) -> Grid:
    r_edges = np.linspace(0.0, R, Nr+1)
    z_edges = np.linspace(Zmin, Zmax, Nz+1)
    r_c = 0.5*(r_edges[:-1] + r_edges[1:])
    z_c = 0.5*(z_edges[:-1] + z_edges[1:])
    dr = np.diff(r_edges)
    dz = np.diff(z_edges)
    Nr_, Nz_ = Nr, Nz
    rP  = r_c[:, None] * np.ones((Nr_, Nz_))
    dr_ = dr[:, None]  * np.ones((Nr_, Nz_))
    dz_ = np.ones((Nr_, Nz_)) * dz[None, :]
    V   = (2*np.pi) * rP * dr_ * dz_
    A_e = (2*np.pi) * r_edges[1:, None] * dz_
    A_w = (2*np.pi) * r_edges[:-1, None] * dz_
    A_n = (2*np.pi) * rP * dr_
    A_s = (2*np.pi) * rP * dr_
    return Grid(Nr, Nz, R, Zmin, Zmax, r_edges, z_edges, r_c, z_c, dr, dz, A_e, A_w, A_n, A_s, V)

def assemble_poisson_matrix(grid: Grid, rho: float) -> csr_matrix:
    Nr, Nz = grid.Nr, grid.Nz
    def lin(i,j): return i + Nr*j
    Nc = Nr*Nz
    Ap = lil_matrix((Nc, Nc), dtype=float)
    dr_e = np.empty(Nr); dr_e[:-1] = grid.r_c[1:] - grid.r_c[:-1]; dr_e[-1] = np.nan
    dr_w = np.empty(Nr); dr_w[1:]  = grid.r_c[1:] - grid.r_c[:-1]; dr_w[0]  = np.nan
    dz_n = np.empty(Nz); dz_n[:-1] = grid.z_c[1:] - grid.z_c[:-1]; dz_n[-1] = np.nan
    dz_s = np.empty(Nz); dz_s[1:]  = grid.z_c[1:] - grid.z_c[:-1]; dz_s[0]  = np.nan
    for j in range(Nz):
        for i in range(Nr):
            row = lin(i,j); vol = grid.V[i,j]
            if i < Nr-1:
                ae = (grid.A_e[i,j]) / (rho * dr_e[i])
                Ap[row, lin(i+1,j)] += -ae/vol; Ap[row, row] += ae/vol
            if i > 0:
                aw = (grid.A_w[i,j]) / (rho * dr_w[i])
                Ap[row, lin(i-1,j)] += -aw/vol; Ap[row, row] += aw/vol
            if j < Nz-1:
                an = (grid.A_n[i,j]) / (rho * dz_n[j])
                Ap[row, lin(i,j+1)] += -an/vol; Ap[row, row] += an/vol
            if j > 0:
                aS = (grid.A_s[i,j]) / (rho * dz_s[j])
                Ap[row, lin(i,j-1)] += -aS/vol; Ap[row, row] += aS/vol
    return Ap.tocsr()

def divergence_from_face_fluxes(u_r: Array, u_z: Array, grid: Grid) -> Array:
    Nr, Nz = grid.Nr, grid.Nz
    div = np.zeros((Nr, Nz))
    for j in range(Nz):
        for i in range(Nr):
            Phi_e = u_r[i+1, j] * grid.A_e[i, j]
            Phi_w = u_r[i,   j] * grid.A_w[i, j]
            Phi_n = u_z[i, j+1] * grid.A_n[i, j]
            Phi_s = u_z[i, j  ] * grid.A_s[i, j]
            div[i,j] = (Phi_e - Phi_w + Phi_n - Phi_s) / grid.V[i,j]
    return div

def correct_face_velocities(u_r_star: Array, u_z_star: Array, p: Array,
                            grid: Grid, rho: float, dt: float):
    Nr, Nz = grid.Nr, grid.Nz
    u_r_new = u_r_star.copy()
    for j in range(Nz):
        for i_face in range(1, Nr):
            iW = i_face - 1; iE = i_face
            dr_face = grid.r_c[iE] - grid.r_c[iW]
            dpdr = (p[iE, j] - p[iW, j]) / dr_face
            u_r_new[i_face, j] = u_r_star[i_face, j] - (dt / rho) * dpdr
    u_z_new = u_z_star.copy()
    for i in range(Nr):
        for j_face in range(1, Nz):
            jS = j_face - 1; jN = j_face
            dz_face = grid.z_c[jN] - grid.z_c[jS]
            dpdz = (p[i, jN] - p[i, jS]) / dz_face
            u_z_new[i, j_face] = u_z_star[i, j_face] - (dt / rho) * dpdz
    return u_r_new, u_z_new

def anchor_pressure(Ap: csr_matrix, b: np.ndarray, idx: int = 0):
    Ap = Ap.tolil()
    Ap.rows[idx] = [idx]
    Ap.data[idx] = [1.0]
    Ap = Ap.tocsr()
    b[idx] = 0.0
    return Ap, b

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Nr, Nz = 20, 30
    R, Zmin, Zmax = 0.1, 0.0, 0.3
    rho, g = 1000.0, 9.81
    dt = 0.05
    grid = build_grid(Nr, Nz, R, Zmin, Zmax)
    u_r_star = np.zeros((Nr+1, Nz))
    u_z_star = np.zeros((Nr,   Nz+1))
    u_z_star[:, 1:-1] = -g * dt
    Ap = assemble_poisson_matrix(grid, rho)
    div_u_star = divergence_from_face_fluxes(u_r_star, u_z_star, grid)
    b = -(div_u_star.flatten(order='F')) / dt
    Ap, b = anchor_pressure(Ap, b, idx=0)
    p = spsolve(Ap, b).reshape((Nr, Nz), order='F')
    u_r_new, u_z_new = correct_face_velocities(u_r_star, u_z_star, p, grid, rho, dt)
    u_r_new[0,  :] = 0.0; u_r_new[-1, :] = 0.0
    u_z_new[:,  0] = 0.0; u_z_new[:, -1] = 0.0
    div_u_new = divergence_from_face_fluxes(u_r_new, u_z_new, grid)
    max_div = np.max(np.abs(div_u_new))
    i_mid = Nr//2; z = grid.z_c; p_col = p[i_mid, :]
    Afit = np.vstack([np.ones_like(z), z]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(Afit, p_col, rcond=None)
    expected_slope = -rho*g
    print(f"Max cell divergence after projection: {max_div:.3e} 1/s")
    print(f"Fitted dp/dz = {b_fit:.6e} Pa/m   vs expected = {expected_slope:.6e} Pa/m")
    plt.figure(figsize=(7,5))
    plt.plot(z, p_col/1e5, 'o', label='Numerical p(z) (mid-radius)')
    plt.plot(z, (a_fit + b_fit*z)/1e5, '-', label=f'Linear fit slope = {b_fit/1e5:.3f} bar/m')
    plt.plot(z, (a_fit + expected_slope*z)/1e5, '--', label=f'Expected slope = {expected_slope/1e5:.3f} bar/m')
    plt.xlabel('z (m)'); plt.ylabel('Pressure (bar)')
    plt.title('Hydrostatic check after projection (staggered grid)')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    from plot_utils_rz import plot_pressure_rz

    plot_pressure_rz(p, grid.r_edges, grid.z_edges, title="Pressure (hydrostatic test)")
