import sys
sys.path.append('..')

from matplotlib import pyplot as plt

from Vec2d import Vec2d
import numpy as np

from Data import Data

#########################################
class Material:
    def __init__(self, P=0.0, u = 0.0, name = None):
        self.P = P
        self.u = u
        self.name = name

#########################################

class Background:
    def __init__(self, mesh, P = None, u = None):
        self.mesh = mesh
        self.P = P
        self.u = u

    def plotIt(self, plot='P'):
        if plot == 'P':
            f = self.P
        else:
            f = self.u
            
        fig, ax = plt.subplots()
        x,y = self.mesh.x, self.mesh.y
        cm = plt.pcolormesh(x, y, f, shading='nearest')
        cm.set_array(f.flatten())

        ax.set_aspect('equal')
        plt.xlabel('cm')
        plt.ylabel('cm')
        plt.title('%s' % plot)
        plt.show()
        
            

#########################################

class Grid:
    def __init__(self, Lx = 1.0, Ly = 1.0, Nx = 10, Ny = 10):
        self.nx = Nx
        self.ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx/Nx
        self.dy = Ly/Ny
        self.x, self.y = np.meshgrid( np.linspace(0,Lx,self.nx+1), np.linspace(0,Ly,self.ny+1), indexing='ij' )
        print('len(self.x) = ', len(self.x))
        print('len(self.x[0]) = ', len(self.x[0]))
        # self.nodes = []
        # for i in range(Nx+1):
        #     for j in range(Ny+1):
        #         self.nodes.append(Node(Vec2d(i*self.dx,j*self.dy),i,j))
        return

#########################################

def backgroundFromBox(mesh, lowerLeft, upperRight, material=None):
    '''makes a numpy array that fits on the given mesh with values in the box and zeros elsewhere'''
    P = np.zeros([mesh.nx, mesh.ny])
    u = np.zeros([mesh.nx, mesh.ny])
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            pos = Vec2d(i*mesh.dx,j*mesh.dy)
            if pos.x >= lowerLeft.x and pos.x <= upperRight.x and \
               pos.y >= lowerLeft.y and pos.y <= upperRight.y:
                P[i, j] = material.P
                u[i, j] = material.u

    newBkg = Background(mesh, P = P, u = u)
    return newBkg

#########################################
def backgroundFromCircle(mesh, center=Vec2d(0,0), radius=1.0, material=None):
    '''makes a numpy array that fits on the given mesh with material values in the circle and zeros elsewhere'''
    P = np.zeros([mesh.nx, mesh.ny])
    u = np.zeros([mesh.nx, mesh.ny])
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            pos = Vec2d(i*mesh.dx,j*mesh.dy)
            r2 = (pos - center).magnitude2
            if r2 < radius**2:
                P[i, j] = material.P
                u[i, j] = material.u
                
    newBkg = Background(mesh, P = P, u = u)
    return newBkg

#########################################

class Node:
    def __init__(self, position, i, j):
        self.pos = position
        self.i = i
        self.j = j
        
#########################################

def overlay_makeBackground(mesh, bkgList):
    '''each non-zero entry in a bkg Field in the list overwrites the
    previous ones---good for building up a complicated geometry via
    painting
    '''
    newBkg = Background(mesh, P = bkgList[0].P, u = bkgList[0].u)
    for b in bkgList[1:]:
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                if b.P[i, j] > 0:
                    newBkg.P[i, j] = b.P[i, j]
                    newBkg.u[i, j] = b.u[i, j]
                
    return newBkg
#########################################
