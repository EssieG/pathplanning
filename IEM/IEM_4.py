"""
Created on Wed Jan  14 18:37:32 2021

Solving with pulse basis functions. start/goal represented as boxes so that they have a normal derivative.
New improvements.

For Behavior 4 of paper, we specify gamma (normal derivative) on all outer boundaries and solve for u (the potential).
At the start and end points we specify the potential and calculate the normal derivatives.

@author: estherg
"""

import numpy as np
import numpy.linalg as la
import math
pi = math.pi
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

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

def intDgreens(p,q1,q2,location):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                  # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                  # w is other variable, held constant over segment 
    Q = np.subtract(q2,q1)
    m = np.zeros(2); m[w]=1; m[v] = 0  #positive normal vector
    if location == 'b':                #outer boundary normal
        nw = (1 if np.cross(Q,m) > 0 else -1)
    else:                              # start and goal normal
        nw = (-1 if np.cross(Q,m) > 0 else 1)
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
flag = [0,0,0]
lseg_sg = 0.1 #length of start/goal segments SPECIFIC
lseg = 0.5 #length of boundary segments SPECIFIC

# Discretize outer boundary
bounds = make_box(0.5, 30, 20) #coordinates of discretized domain, SPECIFIC 1.5X1 with s/g .1x.1
start = make_box(0.1, 10, 10, [2, 2])   #SPECIFIC
goal = make_box(0.1, 10, 10, [12,7])  #SPECIFIC

boundsA = np.array(bounds)
start_u = 1 #start potential
goal_u = 0 #goal potential
sA=np.array(start)
gA=np.array(goal)
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
p = []; s=[]; g=[] #M outer box point coordinates, p contains all from boundary,start, and goal
C_Q = [1/2 for i in bounds];
[C_Q.append(1/2) for i in start];
[C_Q.append(1/2) for i in goal];
p = [list_avg(bounds[i-1],bounds[i]) for i in range(len(bounds))]
s = [list_avg(start[i-1],start[i]) for i in range(len(start))] #start goal boundary points
g = [list_avg(goal[i-1],goal[i]) for i in range(len(goal))]   #goal boundary points
gam_Q = [0 for i in bounds]
[gam_Q.append(1) for i in start] #for start and goal point, unknown
[gam_Q.append(1) for i in goal]
u_Q = [1 for i in bounds]
[u_Q.append(start_u) for i in start] #for start point
[u_Q.append(goal_u) for i in goal]  #for goal point
[p.append(i) for i in s]; 
[p.append(i) for i in g];

#set up matrix A (unknowns)
M = len(p); G=len(g); S=len(s); B=len(bounds) #p is all boundary points
#M = B; G=0; S=0 #TESTING
A = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        if j >= B:       #start and goal positions where potential is known
            if j>= B+S:
                A[i,j] = -intgreens(p[i], goal[j-B-S-1], goal[j-B-S])
            else:
                A[i,j] = -intgreens(p[i], start[j-B-1], start[j-B])
        else:
            A[i,j] = intDgreens(p[i], bounds[j-1], bounds[j], 'b')
            if i == j:
                A[i,j] += C_Q[i]
                
#set up matrix b (knowns)
b = np.zeros(M)
for i in range(M):
    for j in range(M):
        if j>= B:
            if j>= B+S:
                b[i] += -u_Q[j]*intDgreens(p[i], goal[j-B-S-1], goal[j-B-S], 'g')
            else:
                b[i] += -u_Q[j]*intDgreens(p[i], start[j-B-1], start[j-B], 's')
            if i == j:
                b[i] += -u_Q[j]*C_Q[j]
        else:
            b[i] += gam_Q[j]*intgreens(p[i],bounds[j-1],bounds[j])
            
   
# Solve matrix equation and place unknown U and gam in their respective arrays. 
coef = la.solve(A,b)
for i in range(M):  # M unknowns were solved for
    if i < M-S-G:
        u_Q[i] = coef[i]
    else:
        gam_Q[i] = coef[i]

# Plot solution to the potential on the boundary        
fig = plt.figure()
ax = Axes3D(fig)
pA = np.array(p[0:M-S-G])
ax.plot(list(pA[:,0]), list(pA[:,1]), u_Q[0:M-S-G], color='blue')
plt.plot(boundsA[:,0],boundsA[:,1], 'ro')
plt.plot(sA[:,0], sA[:,1], 'ro')
plt.plot(gA[:,0],gA[:,1], 'ro')