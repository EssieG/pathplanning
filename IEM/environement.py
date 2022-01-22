
"""
Script: IEM6.py
Author: Esther Grossman
Date: 3/22/21

Solving with pulse basis functions. The start and goal points are ON the boundary.
Potential specified EVERYWHERE. Normal derivative calculated on all boundaries
and start/goal position.

"""

import ipdb
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
    if p[w] == q1[w] and round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms
        func = lambda x : -1/(pi)*np.lib.scimath.log(np.lib.scimath.sqrt((p[v]-x)**2+(p[w]-q1[w])**2))
        Uint = scipy.integrate.quad(func, p[v], q2[v])
    else:  
        func = lambda x : -1/(2*pi)*np.lib.scimath.log(np.lib.scimath.sqrt((p[v]-x)**2+(p[w]-q1[w])**2))
        Uint = scipy.integrate.quad(func, q1[v], q2[v])
    Uint = Uint[0]
    return Uint

def intDgreens(p,q1,q2, ext = True):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                  # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                  # w is other variable, held constant over segment 
    Q = np.subtract(q2,q1)
    m = np.zeros(2); m[w]=1;  #positive normal vector
    nw = (-1 if np.cross(Q,m) > 0 else 1)  #outer boundary normal pointing outward from walls 
    if ext == False:
        nw = -nw
    if p[w] == q1[w]: 
        if round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms solution
            if q2[v]>q1[v]:
                return -1/2
            else:
                return 1/2
    dfunc = lambda x : (p[w]-q1[w])/(2*pi*((p[w]-q1[w])**2+(p[v]-x)**2))*nw #gradient of greens function with respect to w (normal direction)
    DUint = scipy.integrate.quad(dfunc, q1[v], q2[v]) 
    DUint = DUint[0]
    return DUint

def intDgreensA(p,q1,q2):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                   # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                  # w is other varaible, held constant over segment 
    Q = np.subtract(q2,q1)
    m = np.zeros(2); m[w]=1;  #positive normal vector
    nw = (-1 if np.cross(Q,m) > 0 else 1)  #outer boundary normal pointing outward
    if p[w] == q1[w]:   #p and q on same border
        if round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2):
            if q2[v] > q1[v]:
                return -1/2   # self terms
            else:
                return +1/2
        else:
            return 0      #border terms
    else:
        DUinta = -1/(2*pi)*math.atan((p[v]-q1[v])/(p[w]-q1[w]))
        DUintb = -1/(2*pi)*math.atan((p[v]-q2[v])/(p[w]-q2[w]))
    DUint = nw*(DUintb - DUinta)
    return DUint


#============================================================================
    # MAIN CODE #
#============================================================================
# Set Parameters
intgreens.counter1 = 0;  intgreens.counter2 = 0;  intgreens.counter3 = 0  
lseg=.5     #SPECIFIC length of each segment
nx = 11     #SPECIFIC number of points on x (odd number)
ny = 15     #SPECIFIC number of points on y (odd number)
#ipdb.set_trace()

# Discretize outer boundary
obs = make_box(lseg, 3,3, corner = [nx*lseg/3, ny*lseg/3])
bounds = make_box(lseg, nx, ny)  #coordinates of discretized domain
O = len(obs); M = len(bounds); T = O + M #T is total number of discretized points
goal = bounds
bigBounds = np.vstack((bounds,obs))

# Set up boundary conditions for collocation
c_Q = np.full(M,1/2)
p1 = [list_avg(bounds[i-1],bounds[i]) for i in range(M)]
p2 = [list_avg(obs[i-1],obs[i]) for i in range(O)]  
p = np.vstack((p1,p2))
u_Q = np.ones(M+O) #known and unknown potential at boundary point p, obstacles and start set to 0
u_Q[int(M/2)] = 0; #set goal to be halfway around boundary. set equal to zero
gam_Q = np.ones(M+O)
gam_Q[0:M] = 0
u_indices =np.sort(np.concatenate((np.arange(M,T),[0],[int(M/2)]))) #know potential already at these indices of p
gam_indices = np.sort(np.concatenate((np.arange(1,int(M/2)), np.arange(int(M/2+1),M)))) #know derivative already at these indices of p

# Set up matrix A (unknowns) 
A = np.zeros((T,T))
for i in range(T):                     #sum over boundary elements and obstacle elements separately for indexing purposes
    for k in range(M):
        if k in u_indices:    
            A[i,k] = intDgreens(p[i], bounds[k-1], bounds[k])
            if i == k:
                A[i,k] += u_Q[i]*c_Q[i]
        else:           
            A[i,k] = - intgreens(p[i], bounds[k-1], bounds[k])
    for k in range(O):
        if k in u_indices:    
            A[i,M+k] = intDgreens(p[i], obs[k-1], obs[k], False)
            if i == k:
                A[i,M+k] += u_Q[i]*c_Q[i]
        else:           
            A[i,M+k] = - intgreens(p[i], obs[k-1], obs[k])

# Set up matrix b (knowns)
b = np.zeros(T)
for i in range(T):
    for k in range(M):
        if k in u_indices:    
            b[i] += - u_Q[k]*intDgreens(p[i], bounds[k-1], bounds[k])
            if i == k:
                b[i] += - u_Q[i]*c_Q[i]
        else:
            b[i] += gam_Q[k]*intgreens(p[i], bounds[k-1], bounds[k])
    for k in range(O):
        if k in u_indices:    
            b[i] += - u_Q[M+k]*intDgreens(p[i], obs[k-1], obs[k], False)
            if i == k:
                b[i] += - u_Q[i]*c_Q[i]
        else:
            b[i] += gam_Q[M+k]*intgreens(p[i], obs[k-1], obs[k])

   
# Solve matrix equation 
coef = la.solve(A,b)   #coeff are all the values of gam
gam_Q[u_indices] = coef[u_indices]
u_Q[gam_indices] = coef[gam_indices]


#===========================================================================
    # Generate Plots #
#===========================================================================

# plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients
boundsA = np.array(bigBounds)
fig = plt.figure()
ax = Axes3D(fig)
pA = np.array(p[0:T])
u_bound_est = [] 
for i in range(T):
    temp = 0
    for k in range(M):
        temp += 2*(gam_Q[k]*intgreens(pA[i],bounds[k-1],bounds[k])-u_Q[k]*intDgreens(pA[i], bounds[k-1], bounds[k]))
    for k in range(O):
        temp += 2*(gam_Q[M+k]*intgreens(pA[i],obs[k-1],obs[k])-u_Q[M+k]*intDgreens(pA[i], obs[k-1], obs[k], False))
    u_bound_est.append(temp)
ax.plot(list(pA[:,0]), list(pA[:,1]), u_bound_est[0:T], 'o', color='blue')
plt.plot(boundsA[:,0],boundsA[:,1], 'ro')

# =============================================================================
# # Plot initial potential on the boundary        
# fig = plt.figure()
# ax = Axes3D(fig)
# pA = np.array(p[0:M])
# ax.plot(list(pA[:,0]), list(pA[:,1]), u_Q[0:M], color='blue')
# plt.plot(boundsA[:,0],boundsA[:,1], 'ro')
# =============================================================================

# Solve for points inside the domain and plot potential inside domain
X,Y = np.mgrid[0.1:lseg*nx-0.1:30j, 0.1:lseg*ny-0.1:30j]
xy = np.vstack((X.flatten(), Y.flatten())).T #coordinates of interior as MNx2 vector
phi = np.zeros((len(xy),1))
for i in range(len(xy)):
    for k in range(M):
        phi[i] += 2*(gam_Q[k]*intgreens(xy[i],bounds[k-1],bounds[k])-u_Q[k]*intDgreens(xy[i], bounds[k-1], bounds[k]))
    for k in range(O):
        phi[i] += 2*(gam_Q[M+k]*intgreens(xy[i],obs[k-1], obs[k])-u_Q[M+k]*intDgreens(xy[i], obs[k-1], obs[k], False))
plot3D(xy[:,0],xy[:,1], phi[:,0])       
     


#condition number of matrix
#la.cond(A)















