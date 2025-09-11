# plotting functions for Swirl
# TUE September  9, 2025

import matplotlib.pyplot as plt
import numpy as np
from internal_wall_mask_patch import (
    r_wall_of_z,
)

def _cell_center_velocities(u_r, u_z, grid):
    '''Compute cell-centered (u_r,u_z) by averaging adjacent face values.'''
    Nr, Nz = grid.Nr, grid.Nz
    ur_c = 0.5*(u_r[0:Nr, :] + u_r[1:Nr+1, :])      # average in r
    uz_c = 0.5*(u_z[:, 0:Nz] + u_z[:, 1:Nz+1])      # average in z
    return ur_c, uz_c

def plot_pressure_contours(p, grid, title=None, unit='Pa', scale=1.0,
                           n_contours=16, filename=None, show=False,
                           eta = None, showWall=True):
    '''Pressure contours on (r,z) grid.'''
    Nr, Nz = p.shape
    r_edges, z_edges = grid.r_edges, grid.z_edges
    r_c, z_c = grid.r_c, grid.z_c

    plt.figure(figsize=(7,5))
    pm = plt.pcolormesh(r_edges, z_edges, (p.T)/scale, shading='auto')
    cbar = plt.colorbar(pm); cbar.set_label(f'Pressure [{unit}]')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    CS = plt.contour(r_c, z_c, (p.T)/scale, levels=n_contours, colors='white')
    plt.clabel(CS, inline=True, fontsize=10) # ,levels=[0.0],fmt=f'liquid-air-interface')
    # CS = plt.contour(r_c, z_c, (p.T)/scale, levels=1, colors='red')
    # plt.clabel(CS, inline=True, fontsize=10 ,levels=[0.0],fmt=f'')
    if eta is not None:
        plt.plot(r_c, eta, 'r-', label='liquid-air interface')
    if title: plt.title(title)
    plt.tight_layout()
    if showWall:
        z_vals = np.linspace(grid.wall_zmin, grid.wall_zmax, 100)
        R = grid.r_edges[-1]
        r_vals = r_wall_of_z(R, z_vals)
        plt.plot(r_vals, z_vals, 'r-', lw=2, label='slanted wall')
    
    if filename is not None:
        plt.savefig(filename,dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_pressure_with_swirl(p, u_t, grid, title=None, unit='Pa', scale=1.0, show_contours=True,
                             n_contours=16, filename=None, show=False,
                             eta=None, showWall=True):
    '''Pressure colormap on (r,z) with optional u_theta contours on cell centers.'''
    Nr, Nz = p.shape
    r_edges, z_edges = grid.r_edges, grid.z_edges
    r_c, z_c = grid.r_c, grid.z_c

    plt.figure(figsize=(7,5))
    # pm = plt.pcolormesh(r_edges, z_edges, (p.T)/scale, shading='auto')
    # cbar = plt.colorbar(pm); cbar.set_label(f'u_theta [{unit}]')
    utm = plt.pcolormesh(r_edges, z_edges, (u_t.T), shading='auto')
    cbar = plt.colorbar(utm); cbar.set_label(f'u_theta [{unit}]')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    if show_contours:
        # Contour u_theta on centers
        CS = plt.contour(r_c, z_c, u_t.T, levels=n_contours)
        plt.clabel(CS, inline=True, fontsize=8)
    if eta is not None:
        plt.plot(r_c, eta, 'r-', label='liquid-air interface')
    if showWall:
        z_vals = np.linspace(grid.wall_zmin, grid.wall_zmax, 100)
        R = grid.r_edges[-1]
        r_vals = r_wall_of_z(R, z_vals)
        plt.plot(r_vals, z_vals, 'r-', lw=2, label='slanted wall')
    if title: plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_streamlines(u_r, u_z, grid, density=1.5, title='Streamlines (u_r, u_z)',
                     filename=None, show=False,
                     eta=None, showWall=True):
    '''Streamlines of the meridional velocity field on cell centers.'''
    r_c, z_c = grid.r_c, grid.z_c
    ur_c, uz_c = _cell_center_velocities(u_r, u_z, grid)

    plt.figure(figsize=(7,5))
    # streamplot expects arrays in (x,y) order; our center fields are (Nr,Nz) corresponding to (r,z)
    plt.streamplot(r_c, z_c, ur_c.T, uz_c.T, density=density)
    if eta is not None:
        plt.plot(r_c, eta, 'r-', label='liquid-air interface')
    if showWall:
        z_vals = np.linspace(grid.wall_zmin, grid.wall_zmax, 100)
        R = grid.r_edges[-1]
        r_vals = r_wall_of_z(R, z_vals)
        plt.plot(r_vals, z_vals, 'r-', lw=2, label='slanted wall')
    plt.xlabel('r (m)'); plt.ylabel('z (m)')
    plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

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

def volume_from_pressure_isobar_with_wall(p, grid, masks, patm=0.0, offset=0.0):
    r_c, z_edges = grid.r_c, grid.z_edges
    dr = grid.r_edges[1:] - grid.r_edges[:-1]
    dz = grid.z_edges[1:] - grid.z_edges[:-1]
    Nr = grid.Nr; Nz = grid.Nz
    solid = masks["solid_cell"]  # (Nr,Nz)
    # Recover free-surface height per column from isobar p+offset=patm (as before) → z_fs[i]
    V = 0.0
    for i in range(Nr):
        # find free-surface height in column i by locating sign change in p(i,:)+c - patm
        col = p[i, :] + offset - patm
        # default: completely full/empty cases
        if np.all(col > 0):           # entire column "below" the isobar (full)
            z_fs = z_edges[-1]
        elif np.all(col <= 0):        # entirely above (empty)
            z_fs = z_edges[0]
        else:
            # find the last j with col[j-1]>0 >= col[j] (top-down crossing)
            j = np.argmax(col <= 0)   # first index where <=0 when scanning from bottom; safer to do manual scan if needed
            # robust linear interpolation between (j-1) and j
            j = max(1, min(j, Nz-1))
            p0, p1 = col[j-1], col[j]
            z0, z1 = grid.z_c[j-1], grid.z_c[j]
            t = 0.0 if p1 == p0 else (-p0) / (p1 - p0)
            z_fs = np.clip(z0 + t * (z1 - z0), z_edges[0], z_edges[-1])

        # integrate up to z_fs, but exclude solids
        frac = np.clip((z_fs - z_edges[:-1]) / dz, 0.0, 1.0)  # fraction of each axial cell filled
        frac[solid[i, :]] = 0.0
        V += 2 * np.pi * r_c[i] * dr[i] * np.sum(frac * dz)

    return V

def choose_pressure_offset_for_volume(p, grid, V_target, masks, patm=0.0, c_lo=-1e7, c_hi=+1e7, tol=1e-6):
    # Bisection on offset c so that volume_from_pressure_isobar(p,grid,patm,c)=V_target
    f_lo = volume_from_pressure_isobar_with_wall(p, grid, masks, patm, c_lo) - V_target
    f_hi = volume_from_pressure_isobar_with_wall(p, grid, masks, patm, c_hi) - V_target

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
   pass # no unit test for this yet
    
