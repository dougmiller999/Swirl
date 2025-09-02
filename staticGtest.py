# static pressure versus gravity test 2DRZ

# mesh is 3 r-zones, 10 z-zones

from math import *
# import numpy as np
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve

##### EQUATIONS #####
# We are solving the steady state incompressible RZ Navier-Stokes
# equations.  Actually the first version will probably be zero
# viscosity, so that's just the Euler equations, but let's not be
# pedantic.

# Euler version!
# rho (u dot grad) u = -grad P - rho*g(z^)
# div(u) = 0

# rho div(u u) = -grad P - rho*g(z^), actually better for FV
# div(u) = 0
##### EQUATIONS #####

##### DIFFERENCE EQUATIONS #####
# Lotsa ways to discretize these, we are FV folk.
# Keeping the momentum equation in a div(flux) form
# allows us to use mass flux terms that we can re-use in
# the div(u)=0 condition, making the scheme compatible.

# radial term
# rho ((1/r) (d/dr (r ur ur)) + d/dz(uz ur)) - ut**2/r = -dP/dr
# theta term
# rho ((1/r) (d/dr (r ur ut)) + d/dz(uz ut)) + ur*ut/r = -dP/dt = 0
# z term
# rho ((1/r) (d/dr (r ur uz)) + d/dz(uz uz)) = -dP/dz - rho*g

# div(u) = (1/r) (d/dr (r ur)) + d/dz(uz)) = 0
#
# < div(u) >_z = (1/V_z) sum_f u dot A_f, over faces in zone

# Best expressed as face centered mass flux factors m, where
# m_f = rho u_f dot A_f.  We will use
# m_east/west for radial areas
# m_north/south for z-direction area
#
# A_w_i = r_i 2 pi dz
# A_e_i = r_i+1 dt dz = r_i+1 2 pi dz
# A_n_i = dr * 2 pi r_i+1/2
# A_s_i = dr * 2 pi r_i+1/2 
#
# m_w_i = rho * A_w_i * ur_i
# m_e_i = rho * A_e_i * ur_i+1
# m_n_i = rho * A_n_i * uz_i+1
# m_s_i = rho * A_s_i * uz_i
#
# So the incompressibility condition becomes
# m_w_i + m_e_i + m_n_i + m_s_i = 0
#
# And the momentum equations are:
# div(rho u u)_r = (1/V_z) sum_f m ur - ut**2/r_z = -dP/dr
# div(rho u u)_t = (1/V_z) sum_f m ut + ur*ut/r_z = 0
# div(rho u u)_z = (1/V_z) sum_f m uz = -dP/dz - rho*g
#
# So that ought to give us equations for setting up a
# matrix to solve for {P, ur, ut, uz}.  Yay!
##### DIFFERENCE EQUATIONS #####

##### MAKE MATRIX #####
# Now we will assemble the matrix for these equations, collecting all
# the terms, source terms, and boundary conditions in to a vast system
# that we will express as A * x = b

# Size: we have nr*nz zones, where we center P and ut, and (nr+1)*nz
# vertical faces, upon which we center ur, and (nz+1*nr horizontal
# faces, upon which we center uz.
# 2*nr*nz + (nr+1)*nz + (nz+1)*nr = 4*nr*nz + nr + nzj
nr = 4
nz = 4
N = 4*nr*nz + nr + nz
print(" N = ", N)

# I have no idea if the order we arrange the rows makes any difference
# when it comes to solving the matrix.  Let's ask the web and
# ChatGPT.  Nothing there, but there is a lot of chatter about using
# MKL, which might be faster than what scipy uses.  JUST GET IT
# WORKING FIRST, **THEN** OPTIMIZE, I say this to everyone why can't I listen?

# first, what does our solution vector look like?  We have to put in
# an entry for each unknown (each entry will have an associated row in
# the matrix).
P0 = 0                     # first pressure index
Pn = P0 + nr*nz -1         # last pressure index
ur0 = Pn+1                 # first ur index
urn = ur0 + (nr+1)*nz -1   # last ur index
ut0 = urn+1                # first ut index
utn = ut0 + nr*nz - 1      # last ut index
uz0 = utn+1                # first uz index
uzn = uz0 + (nz+1)*nr - 1  # last uz index

print(" Pn = ", Pn)
print(" ur0 = ", ur0)
print(" urn = ", urn)
print(" ut0 = ", ut0)
print(" utn = ", utn)
print(" uz0 = ", uz0)
print(" uzn = ", uzn)

# soln = np.zeros(N)

# an nr=4, nz=4 grid
# 3 | 12 13 14 15
# 2 |  8  9 10 11
# 1 |  4  5  6  7
# 0 |  0  1  2  3
#    ------------
#      0  1  2  3

# matrix rows for the divergence free condition:
# Constant density makse the incompressibility condition:
# A_w_i*ur_i + A_e_i*ur_i+1 + A_n_i*uz_i+1 + A_s_i*uz_i = 0
# for the i'th zone.

# Zone k has the following indices for it's face velocities:
# u_w(k) = ur0 + (k//nr)*(nr+1) +  k - (k//nr)*nr, which simplifies
# u_w(k) = ur0 + k//nr +  k + 1
k = 14
irw = k//nr +  k
ire = k//nr +  k + 1
izs = k
izn = k + nr
print(" k = ", k)
print(" ire = ", ire)
print(" irw = ", irw)
print(" izs = ", izs)
print(" izn = ", izn)
# 
# So for zone k, we get a row:
# Ar[irw] * ur[irw] + Ar[ire] * ur[ire] + Az[izs] * uz[izs] + Az[izn] * uz[izn] = 0 


# # # Define diagonals
# # main_diag = 2 * np.ones(n)
# # off_diag = -1 * np.ones(n-1)

# # # Construct sparse tridiagonal matrix
# # A = diags(
# #     diagonals=[off_diag, main_diag, off_diag],
# #     offsets=[-1, 0, 1],
# #     format="csr"   # compressed sparse row format
# # )

# # # Right-hand side
# # b = np.array([1, 0, 0, 1], dtype=float)



##### MAKE MATRIX #####

##### SOLVE MATRIX #####

# # # Solve
# # x = spsolve(A, b)

##### SOLVE MATRIX #####

##### PLOT SOLUTION #####
##### PLOT SOLUTION #####




# print("Solution:", x)
