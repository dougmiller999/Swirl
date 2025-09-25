import numpy as np
import sys
sys.path.append('..')

from simple_solver_swirl import grad_p_on_faces
from axisym_poisson_projection import Grid, build_grid
from adjointTest import myGradFD

def run_grad(G, label):

    print(label)
    
    Nr = 3
    Nz = 3
    R = 3.0
    H = 10.0
    Zmin = 0.0
    Zmax = H
    grid = build_grid(Nr, Nz, R, Zmin, Zmax)

    r = grid.r_c  # zone center r
    z = grid.z_c  # zone center z
    rr = (r*(r-R)).reshape((len(r),1)) # make a column vector
    zz = z*(z-H)  # w/o reshape is a row vector
    p = rr * zz # outer product

    Gp_r, Gp_z = G(p, grid)
    print(" Gp_r = ", Gp_r)
    print(" Gp_z = ", Gp_z)

    TGp_r = np.zeros((grid.Nr+1, grid.Nz),dtype=float)
    TGp_z = np.zeros((grid.Nr, grid.Nz+1),dtype=float)
    for i in range(1,grid.Nr):
        r = grid.r_edges[i]
        for j in range(grid.Nz):
            z = grid.z_c[j]
            TGp_r[i,j] = z*(z-H)*(2*r-R)
    for i in range(grid.Nr):
        r = grid.r_c[i]
        for j in range(1,grid.Nz):
            z = grid.z_edges[j]
            TGp_z[i,j] = (2*z-H)*r*(r-R)
    print(" TGp_r = ", TGp_r)
    print(" TGp_z = ", TGp_z)

    errR = np.sum(np.abs(Gp_r - TGp_r))
    errZ = np.sum(np.abs(Gp_z - TGp_z))
    print(" errR = ", errR)
    print(" errZ = ", errZ)

    return errR < 1e-12 and errZ < 1e-12
    

def test_grad_p_on_faces():
    assert(run_grad(grad_p_on_faces, 'grad_p_on_faces()'))

def test_myGrad():
    assert(run_grad(myGradFD, 'myGradFD()'))
