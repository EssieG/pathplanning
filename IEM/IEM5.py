# -*- coding: utf-8 -*-
"""
Script: IEM5.py
Author: Esther Grossman
Date: 2/3/21

Solving with pulse basis functions. The start and goal points are ON the boundary.
Potential specified at goal. Normal derivative specified on all other point boundaries
and start position.

"""

import numpy as np
import numpy.linalg as la
import math
pi = math.pi
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

def plot3D(x,y,z):
    '''Plots anything in 3D. Intended for showing potential surface over 2D plane.'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x,y,z,cmap=cm.coolwarm, linewidth=0,antialiased=False)
    plt.title('IEM grid')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi$')
    plt.show()

def make_box(lseg, Nx, Ny, corner = [0,0]):
    '''Nx is the number of discretized point you want on the x dimension. 1 points on
    each boundary, this would be Nx=1. lseg is the length between points, so the total
    dimension along x would be lseg*Nx. corner is a coordinate(list) that tells where to
    put the lower left corner of the box'''
    x = corner[0]; y=corner[1];
    box = []
    for i in range(Nx):
        box.append([round(x,2),round(y,2)])  #must round because imperfect adding of decimals
        x += lseg
    for i in range(Ny):
        box.append([round(x,2),round(y,2)])
        y += lseg
    for i in range(Nx):
        box.append([round(x,2),round(y,2)])
        x -= lseg
    for i in range(Ny):
        box.append([round(x,2),round(y,2)])
        y -= lseg    
    return box

def list_avg(x1,x2):
    assert isinstance(x1,list)
    '''take two points as lists and return their midpoint as a list'''
    diff = np.subtract(x2,x1)/2
    avg = list(np.round(np.add(diff,x1),2))
    return avg

def dist(x1,x2):
    '''Distance between two points'''
    r = math.sqrt((x2[0]-x1[0])^2+(x2[1]-x1[1])^2)
    return r

def greens(x1,x2):
    '''Input is two cartesian coordinates. Output in Green's potential'''
    U = 1/(2*math.pi)*(math.log(1/dist(x1,x2)))
    return U

def intgreens(p,q1,q2):
    '''Input is reference point p and the segment points to integrate Greens function over. Output is 
    integral of Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                   # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                   # w is other variable, held constant over segment 
    if round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms
        func = lambda x : -1/(pi)*np.lib.scimath.log(np.lib.scimath.sqrt((p[v]-x)**2+(p[w]-q1[w])**2))
        Uint = scipy.integrate.quad(func, p[v], q2[v])
    else:  
        func = lambda x : -1/(2*pi)*np.lib.scimath.log(np.lib.scimath.sqrt((p[v]-x)**2+(p[w]-q1[w])**2))
        Uint = scipy.integrate.quad(func, q1[v], q2[v])
    Uint = Uint[0]
    return Uint

def intDgreens(p,q1,q2):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                  # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                  # w is other variable, held constant over segment 
    Q = np.subtract(q2,q1)
    m = np.zeros(2); m[w]=1; m[v] = 0  #positive normal vector
    nw = (-1 if np.cross(Q,m) > 0 else 1)  #outer boundary normal
    if p[w] == q1[w] and round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms solution
        intgreens.counter1 += 1
        if q2[v]>q1[v]:
            return nw*1/2
        else:
            return nw*-1/2
    dfunc = lambda x : nw*(p[w]-q1[w])/(2*pi*((p[w]-q1[w])**2+(p[v]-x)**2)) #gradient of greens function with respect to w (normal direction)
    DUint = scipy.integrate.quad(dfunc, q1[v], q2[v]) 
    #print(DUint)
    intgreens.counter2 += 1
    DUint = DUint[0]
    return DUint


#=============================================================================
    # MAIN CODE #
#=============================================================================
intgreens.counter1 = 0  
intgreens.counter2 = 0
intgreens.counter3 = 0  
lseg=.5   #SPECIFIC
nx = 30     #SPECIFIC
ny = 20     #SPECIFIC

# Discretize outer boundary
bounds = make_box(lseg, nx, ny) #coordinates of discretized domain, SPECIFIC 1.5X1 with s/g .1x.1
start = [0.0, 2.25]  #SPECIFIC
goal = [15.0, 7.25]  #SPECIFIC

boundsA = np.array(bounds)
start_u = 1 #start potential
goal_u = 0 #goal potential

# Plot outer boundary
# =============================================================================
# plt.figure(1)
# plt.plot(boundsA[:,0],boundsA[:,1], 'ro')
# plt.plot(sA[:,0], sA[:,1], 'ro')
# plt.plot(gA[:,0],gA[:,1], 'ro')
# =============================================================================

# Set up integral equations for empty boundary
C_Q = [] #smoothness factor for points p
gam_Q = [] #known and unknown derivative at boundary point p
u_Q  = [] #known and unknown potential at boundary point p
C_Q = [1/2 for i in bounds];
p = [list_avg(bounds[i-1],bounds[i]) for i in range(len(bounds))]
gam_Q=[]
for i in p:
    if i == start:
         gam_Q.append(1)
         u_Q.append(1)
    elif i == goal:
         print(i)
         gam_Q.append(1)
         u_Q.append(0)
    else:
         gam_Q.append(0)
         u_Q.append(1)
         
#set up matrix A (unknowns)
M = len(p)           #p is all boundary points
A = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        if p[j] == start or p[j] == goal:       #start and goal positions where potential is known
                A[i,j] = -intgreens(p[i], bounds[j-1], bounds[j])
        else:
            A[i,j] = intDgreens(p[i], bounds[j-1], bounds[j])
            if i == j:
                A[i,j] += C_Q[i]
                
#set up matrix b (knowns)
b = np.zeros(M)
for i in range(M):
    for j in range(M):
        if p[j] == start or p[j] == goal: 
            b[i] += -u_Q[j]*intDgreens(p[i], bounds[j-1], bounds[j])
            if i == j:
                b[i] += -u_Q[j]*C_Q[j]
        else:
            b[i] += gam_Q[j]*intgreens(p[i],bounds[j-1],bounds[j])
            
   
# Solve matrix equation and place unknown U and gam in their respective arrays. 
coef = la.solve(A,b)
for i in range(M):  # M unknowns were solved for
    if p[i] == start or p[i] == goal:
        gam_Q[i] = coef[i]
        print(i)
    else:
        u_Q[i] = coef[i]

# Plot solution to the potential on the boundary        
fig = plt.figure()
ax = Axes3D(fig)
pA = np.array(p[0:M])
ax.plot(list(pA[:,0]), list(pA[:,1]), u_Q[0:M], color='blue')
plt.plot(boundsA[:,0],boundsA[:,1], 'ro')

#solve for points inside the domain
X,Y = np.mgrid[0.1:lseg*nx-0.1:5j, 0.1:lseg*ny-0.1:5j]
xy = np.vstack((X.flatten(), Y.flatten())).T #coordinates of interior as MNx2 vector
phi = np.zeros((len(xy),1))
for i in range(len(xy)):
    for j in range(M):
        phi[i] += gam_Q[j]*intgreens(xy[i],bounds[j-1],bounds[j])-u_Q[j]*intDgreens(xy[i], bounds[j-1], bounds[j])
plot3D(xy[:,0],xy[:,1], phi[:,0])       
        











     
# =============================================================================
# #How to elementwise multipy a 3D matrix by a 1D vector
# a=np.ones((12,256,256))
# b=np.array(range(12))
# c=a*b[:,np.newaxis,np.newaxis]
# =============================================================================
