import numpy as np
import sys
sys.path.append('..')

from axisym_poisson_projection import Grid, build_grid, divergence_from_face_fluxes
from adjointTest import myDiv

def run_div(D, label):

    print(label)

    prob_type = "non"
    prob_type = "linear"
    
    Nr = 3
    Nz = 3
    R = 3.0
    H = 10.0
    Zmin = 0.0
    Zmax = H
    grid = build_grid(Nr, Nz, R, Zmin, Zmax)
    
    Fr = np.zeros((grid.Nr+1, grid.Nz),dtype=float)   # radial face test field
    Fz = np.zeros((grid.Nr,   grid.Nz+1),dtype=float) # axial face test field
    TDivF = np.zeros((grid.Nr, grid.Nz),dtype=float)  # theoretical value

    if prob_type != 'linear':
        # nonlinear case, will show discretization error
        for i,r in enumerate(grid.r_edges):
            for j,z in enumerate(grid.z_c):
                Fr[i,j] = r*(r-R)
        for i,r in enumerate(grid.r_c):
            for j,z in enumerate(grid.z_edges):
                Fz[i,j] = z*(z-H)

        for i in range(grid.Nr):
            r = grid.r_c[i]
            for j in range(grid.Nz):
                z = grid.z_c[j]
                TDivF[i,j] = 3*r - 2*R + 2*z - H
        # print(" nonlinear TDivF = ", TDivF)

    else:
        # linear field should get const divergence
        for i,r in enumerate(grid.r_edges):
            for j,z in enumerate(grid.z_c):
                Fr[i,j] = r
        for i,r in enumerate(grid.r_c):
            for j,z in enumerate(grid.z_edges):
                Fz[i,j] = z
            
        TDivF[:,:] = 3
        # print(" linear TDivF = ", TDivF)
        
    divF = D(Fr, Fz, grid)   # your divergence_from_face_fluxes (no extra factors)
    # print(" divF = ", divF)




    err = np.sum(np.abs(divF - TDivF))/(Nr*Nz)
    print(" err = ", err)

    return err < 1e-12
    

def test_divergence_from_face_fluxes():
    assert(run_div(divergence_from_face_fluxes, 'divergence_from_face_fluxes()'))

if __name__ == "__main__":
    run_div(divergence_from_face_fluxes, 'divergence_from_face_fluxes()')
