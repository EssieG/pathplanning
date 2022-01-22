# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:37:32 2021

Solving with pulse basis functions. start/goal represented as 1 dimensional segments.

For Behavior 4 of paper, we specify gamma (normal derivative) on all outer boundaries and solve for u (the potential).
At the start and end points we specify the potential and calculate the normal derivatives.

@author: estherg
"""

import numpy as np
import numpy.linalg as la
import math
pi = math.pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

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
    w = (1 if q1[1]==q2[1] else 0)                  # w is other variable, held constant over segment 
    #integration over straight line segment
    Uinta = 1/2*(p[v]-q1[v])*math.log((p[v]-q1[v])**2+(p[w]-q1[w])**2)+q1[v]
    Uintb = 1/2*(p[v]-q2[v])*math.log((p[v]-q2[v])**2+(p[w]-q2[w])**2)+q2[v]
    if p[w] != q1[w]:    #when p and q dont lie on the same line
        Uinta += (p[w]-q1[w])*math.atan((p[v]-q1[v])/(p[w]-q1[w]))
        Uintb += (p[w]-q2[w])*math.atan((p[v]-q2[v])/(p[w]-q2[w]))
    Uint = 1/(2*pi)*(Uintb - Uinta)
    return Uint

def intDgreens(p,q1,q2):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                  # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                  # w is other variable, held constant over segment 
    if p[w] == q1[w]:   #p and q on same border/line
        if round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms
            if q2[v] > q1[v]:
                DUint=1/2
            else:
                DUint=-1/2 
        else:
            return 0     
    else:
        DUinta = -1/(2*pi)*math.atan((p[v]-q1[v])/(p[w]-q1[w]))
        DUintb = -1/(2*pi)*math.atan((p[v]-q2[v])/(p[w]-q2[w]))
        DUint = DUintb - DUinta
    if q2[w] == 1.5 or q2[w] == 1:    #SPECIFIC parameters for this environment
        intDgreens.counter += 1
        return DUint
    else: 
        intgreens.counter +=1
        return -DUint

#=============================================================================
    # MAIN CODE #
#=============================================================================
intDgreens.counter = 0  
intgreens.counter = 0  
lseg = 0.5  #discretized length
lseg_sg = 0.1 #length of start/goal segments

# Discretize outer boundary
bounds = [] #coordinates of discretized domain, (1X1.5 in dimension). These are also the coordinates of all
       #out points p. All are spaced 0.5 apart. additionally, they are ordered (bounds[i]-bounds[i+1]=0.5)
x = 0
y = 0
for i in range(3):
    bounds.append([round(x,2),round(y,2)])  #must round because imperfect adding of decimals
    x += lseg
for i in range(2):
    bounds.append([round(x,2),round(y,2)])
    y += lseg
for i in range(3):
    bounds.append([round(x,2),round(y,2)])
    x -= lseg
for i in range(2):
    bounds.append([round(x,2),round(y,2)])
    y -= lseg    
boundsA = np.array(bounds)
start = [0.3, 0.3] #start coordinates
start_u = 1 #start potential
goal = [1.2, .7] #goal coordinates
goal_u = 0  #goal potential

# Plot outer boundary
# =============================================================================
# plt.figure(1)
# plt.plot(boundsA[:,0],boundsA[:,1], 'ro')
# plt.plot(start[0],start[1], 'ro')
# plt.plot(goal[0],goal[1], 'ro')
# =============================================================================
#plt.plot(ob1[:,0],ob1[:,1], 'ro')

# Set up integral equations for empty boundary
C_Q = [] #smoothness factor for points p
gam_Q = [] #known and unknown derivative at boundary point p
u_Q  = [] #known and unknown potential at boundary point p
p = [] #M outer box point coordinates 
C_Q = [1/2 for i in bounds]
C_Q.append(1); C_Q.append(1); #start/goal position contained entirely in space
p = [list_avg(bounds[i-1],bounds[i]) for i in range(len(bounds))]
gam_Q = [0 for i in bounds]
gam_Q.append(1); gam_Q.append(1); #for start and goal point, unknown
u_Q = [1 for i in bounds]
u_Q.append(start_u) #for start point
u_Q.append(goal_u)  #for goal point
p.append(start); p.append(goal);

#set up matrix A (unknowns)
M = len(p)
A = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        if j >= M-2:       #start and goal positions where potential is known
            A[i,j] = -intgreens(p[i], list(np.subtract(p[j],[0.05, 0])), list(np.add(p[j],[.05, 0]))) #make little segments for start and goal
        else:
            A[i,j] = intDgreens(p[i], bounds[j-1], bounds[j])
            if i == j:
                A[i,j] += C_Q[i]
                
#set up matrix b (knowns)
b = np.zeros(M)
for i in range(M):
    for j in range(M):
        if j>= M-2:
            b[i] += -u_Q[j]*intDgreens(p[i], list(np.subtract(p[j],[0.05, 0])), list(np.add(p[j],[0.05, 0])))
            if i == j:
                b[i] += -u_Q[j]*C_Q[j]
        else:
            b[i] += gam_Q[j]*intgreens(p[i],bounds[j-1],bounds[j])
            
   
# Solve matrix equation and place unknown U and gam in their respective arrays. 
coef = la.solve(A,b)
for i in range(M):  # M unknowns were solved for
    if i < M-2:
        u_Q[i] = coef[i]
    else:
        gam_Q[i] = coef[i]

# Plot solution to the potential on the boundary        
fig = plt.figure()
ax = Axes3D(fig)
pA = np.array(p[0:M-2])
ax.plot(list(pA[:,0]), list(pA[:,1]), u_Q[0:M-2], color='blue')
plt.show()