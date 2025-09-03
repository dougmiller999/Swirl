# TUE September  2, 2025

# Analytic solution of the surface profile of liquid swirling with
# viscosity in a bucket, under a few different assumptions.

# velocity - (ur, ut, uz), and we imagine a height function h(r)
# We have two equations (let ,x denote partial w.r.t. x):
#
# P,r = rho * ut**2/r
# P,z = -rho g 
#
# Then at the surface P = 1 atm = const, so along the surface,
# dP/dr = 0 = P,r + P,z h,r
# 0 = rho * ut**2 / r  - rho * g * dh/dr
# dh/dr = ut**2 / (g*r)
#
# h(r) = C + int^r{ ut(r')**2 / (g * r') dr'
#
# And we will use a fiat ut(r) and conservation of mass to determine C

from math import *
from matplotlib import pyplot as plt
import numpy as np

R = 3.0 # m
H = 30.0 # m
V = pi*R**2 * H
r = np.linspace(0,R,100)
Omega = 100*2*pi/60.0 # 100 rpm ~ 10.5 rad/sec
g = 9.8 # m/s^2

# model = "SBR" # "SBR", for solid body rotation
model = "Rankine" # "SBR", for solid body rotation

if model == "SBR":
    title = "fluid doing solid body rotation"
    #### SOLID BODY ROTATION
    # Solid body rotation, ut = Omega * r, then
    #
    # h(r) = C + Omega**2 * r**2 / (2*g)
    #
    # Mass conservation sets the C constant, let
    # H = "mean fill height" = V/(pi*R**2), then
    #
    # V = 2*pi* int_0^R { r * h(r) dr}
    # plug in h(r), solve for C
    # V = 2*pi* int_0^R { r(C + Omega**2 * r**2/(2*g))dr}
    # V = pi*R**2*C + pi*Omega**2 * R**4/(4*g)
    # V/(pi*R**2) = H = C + Omega**2 * R**2/(4*g)
    # C = H - Omega**2 * R**2/(4*g)
    #
    # and finally,
    # h(r) = H + Omega**2/(2*g) * (r**2 - R**2/2)
    h = H + Omega**2/(2*g) * (r**2 - R**2/2)
    ###############################

elif model == "Rankine": # Rankine vortex
    title = "fluid doing Rankine vortex"
    #### Rankine Vortex assumption
    # ut = Omega * r,   r <= a
    #      Omega*a^2/r, r > a
    #
    # h(r) = C + Omega**2 * r**2 / (2*g)    , r <=a
    # h(r) = D - Omega**2 * a**4/(2*g*r**2) , r > a
    #
    # continuity at a, => D = C + 2*a**2/L, 1/L = Omega**2/(2*g)
    #
    # Mass conservation sets the C constant, let
    # H = "mean fill height" = V/(pi*R**2), then after some algebra
    #
    # C = H*R**2 - (a**4/L)*(0.5*(1-(r0/a)**4) + 2*((R/a)**2-1) - 2*log(R/a))/
    #               (R**2 - r0**2)
    #
    # and finally,
    # h(r) = C + r**2/L,                    r <= a
    # h(r) = C + 2*a**2/L - a**4/(L*r**2),  r > a
    ###############################

    H = 0.4
    L = Omega/(2*g)
    a = R/3
    r0 = R/4
    C =  H*R**2 - (a**4/L)*(0.5*(1-(r0/a)**4) + 2*((R/a)**2-1) - 2*log(R/a))/(R**2 - r0**2)
    h = np.where(r <= a, C + r**2/L, C + 2*a**2/L - a**4/(L*r**2))
    
plt.plot(r,h, '-', label="liquid-air interface")
plt.xlabel("r (meters)")
plt.ylabel("z (meters)")
plt.plot([a,a],[0.5,3.5], '--', label="r=a")
plt.grid()
plt.legend()
plt.title(title)
plt.show()
