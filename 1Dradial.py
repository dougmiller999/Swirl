# FRI August 29, 2025

# The assumptions here are swirling liquid, no gravity, we are testing
# whether we can get a solution to the viscous steady state Navier-Stokes
# equations that matches the analytic.

# The real goal is to re-acquaint me with the details of numpy linear
# solvers (unless this turns out to be harder than expected, in which
# case the goal will be to learn whatever I need to know before going
# to 2D).  

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import *
import sys

if len(sys.argv) > 1:
    graph = True
else:
    graph = False

# Set print options for arrays
np.set_printoptions(precision=3, formatter={'float':'{:.3e}'.format})

############ SET UP ############################
# hard coding our initial conditions!  Yup, this is a one-off.
# Units are MKS.

N = 3 # resolution
N = 100 # resolution

rho = 450 # Kg/m
sigma = 0.3 # surface tension, in N/m

# we are in (r,theta,z) coordinates, but in 1D there is no z, and we
# will be using _t for _theta, because we don't have time in this
# problem, it's all steady state.

Rwall = 3.0 # m, sets the length scale of our problem
u_t_wall = Rwall * 100*2*pi/60 # m/s, 100 rpm, a number I just made up
Pwall = 101325 * 5 # 5 atmospheres in N/m^2 (Pascals), a guess
Pwall = rho*u_t_wall**2 # pressure in N/m^2 (Pascals)
print('Rwall = %12.5e m' % (Rwall))
print('Pwall = %12.5e N/m^2 = %12.5e atm' % (Pwall,Pwall/101325))
print('u_t_wall = %12.5e m/s = %12.5e rpm' % (u_t_wall,60*u_t_wall/(Rwall*2*pi)))

# Now, it's true that u_t_wall and Pwall must be related but I don't
# know what the relationship is yet.  But since the units are the same
# for P and rho*u^2, I'm going to set Pwall = rho*u_t_wall and see
# what happens.  And wow, at 100 rmp, we got 4.3 atm, disturbingly
# close to my 5 atm guess.
############ SET UP ############################


############ DIFFERENCE EQNS ############################
# the N-S equations, steady state, are
# rho(u dot div) u = - grad(P) + div( mu (grad(u) + grad(u)^T)) + rho*g*z^
#
# 1D radial, with u_t allowed (but not u_r or u_z, cuz steady state 1D),
# reduces to these two eqns:
#
# rho u_t^2/r = dP/dr
# d^2 u_t/dr^2 + (1/r) d u_t/dr - u_t/r = 0

# discretizing

# P_i+1/2 - P_i-1/2 = dr rho_i * u_t_i / r_i

# (u_t_i+1 - 2*u_t_i + u_t_i-1)/dr^2 +
# (1/r_i) * (u_t_i+1 - u_t_i-1)/(2*dr) - u_t_i/r_i^2 = 0

P = np.zeros(N)        # there are N zones
ut = np.zeros(N+1)     # there are N+1 nodes
dr = Rwall/N
r = np.linspace(0,Rwall,N+1)  # there are N+1 nodes
r_z = 0.5*(r[:-1] + r[1:])    # r at mid-zone
#print(r)
#print(r_z)
############ DIFFERENCE EQNS ############################

############ CREATE MATRIX for u_t ######################
# So this 1D is a little weird in that u_t is independent of P.
# We can just solve for u_t first, then solve for P.  So let's
# do that.

# Matrix for solving for u_t:
# for row i we have
# col i-1: (1/dr^2 - 1/(r_i * 2 * dr)) 
# col i: (-2/dr^2 - 1/r_i^2) 
# col i+1: (1/dr^2 + 1/(r_i * 2 * dr))
#
# BC: we know the u_t_wall value, so the last row is
# ( 0 0 0 .....  1)
# and the first row is 
# ( 1 0 0 .....  0)
#
# the RHS is (0 0 0 .... u_t_wall)

# Here is how numpy wants to solve this: make diagonals,
# make the RHS, pass to the solve routine!
main_diag = np.ones(N+1)
main_diag[1:-1] *= -2/dr**2 - 1/r[1:-1]**2
main_diag[0] = 1  # BC
main_diag[-1] = 1 # BC
print("main: ",main_diag)
lower_diag = np.ones(N)
lower_diag[:-1] *= 1/dr**2 - 1/(2*dr*r[1:-1])
lower_diag[-1] = 0
print("lower: ", lower_diag)
upper_diag = np.ones(N)
upper_diag[1:] *= 1/dr**2 + 1/(2*dr*r[1:-1])
upper_diag[0] = 0
print("upper: ", upper_diag)
A = diags(
    diagonals=[lower_diag, main_diag, upper_diag],
    offsets = [-1,0,1],
    format = "csr" # compressed sparse row format
    )

# create right hand side
b = np.zeros(N+1) # recall, there are N+1 nodes
b[-1] = u_t_wall # BC for velocity
############ CREATE MATRIX for u_t ###################

############ SOLVE MATRIX for u_t ####################
u_t = spsolve(A,b)
print("u_t: ", u_t)
############ SOLVE MATRIX for u_t ####################

############ CREATE MATRIX for P ######################
# Now solve for P.
# dP/dr = rho * u_t**2/r
# P_i+1/2 - P_i-1/2 = dr rho_i * u_t_i / r_i
# with BC of Pwall.

# Matrix for solving for P:
# for row i we have
# col i-1: -1
# col i: 1
#
# the RHS is dr*rho*(.... u_t_i/r_i .... Pwall)
# 
# BC: we know the Pwall value, so the last row is
# ( 0 0 0 .....  1)

# Here is how numpy wants to solve this: make diagonals,
# make the RHS, pass to the solve routine!
main_diag = -1*np.ones(N)
main_diag[-1] = 1 # BC
print("main: ",main_diag)
upper_diag = np.ones(N-1)
print("upper: ", upper_diag)
A = diags(
    diagonals=[main_diag,upper_diag],
    offsets = [0,1],
    format = "csr" # compressed sparse row format
    )

# create right hand side
b = np.ones(N) # recall, there are N zones
b[:-1] *= rho*dr*u_t[1:-1]**2/r[1:-1]
omega2 = (u_t_wall/Rwall)**2
Pwall_z = Pwall - 0.5*rho*omega2*(Rwall**2-(Rwall-0.5*dr)**2)
b[-1] = Pwall_z # BC for P
print("b: ", b)
print("b0 = ", dr*rho*u_t[1]/r[1])
print("b1 = ", dr*rho*u_t[2]/r[2])
print("b2 = ", Pwall)
############ CREATE MATRIX for P ###################

############ SOLVE MATRIX for P ####################
P = spsolve(A,b)
print("P: ", P)
Ptheory = Pwall - 0.5*rho*omega2*(Rwall**2 - r_z**2)
print("Ptheory: ", Ptheory)

############ SOLVE MATRIX for P ####################


############ PLOT RESULTS ############################
if graph:
    import matplotlib.pyplot as plt

    x = r
    y = u_t
    y2 = 1 - (x-1)**2

    fig, ax = plt.subplots()

    plt.plot(r,u_t,'o-',label="u")
    plt.plot([0,Rwall],[0,u_t_wall],'--',label="u theory")

    plt.plot(r_z,P/101325,'o-',label="P")
    plt.plot(r_z,Ptheory/101325,'--',label="P theory")

    # a line at 1 atm pressure
    plt.plot([0,Rwall],[101325/101325,101325/101325],'--',label="1 atm")

    # ax.fill(x, y, zorder=10)
    # ax.fill(x, y2, zorder=18,color='white')
    # ax.grid(True, zorder=5)
    plt.title("Pressure and velocity for 1D radial")
    plt.ylabel("P (atm) and u (m/s)")
    plt.xlabel("radius (m)")
    plt.legend()
    
             
    ax.grid(True)
    plt.show()
    
############ PLOT RESULTS ############################

