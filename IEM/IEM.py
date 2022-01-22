# -*- coding: utf-8 -*-

"""
Author: Esther Grossman
IEM : IEM for robot navigation, Original
replication of paper by Mantegh et al. 2010

"""
import numpy as np
import numpy.linalg as la
import math
pi = math.pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.linalg import null_space


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
    w = (1 if q1[1]==q2[1] else 0)                  # w is other varaible, held constant over segment 
    if p == q1:       #one side of corner
        Uinta = 0
        Uintb = 1/2*(p[v]-q2[v])*math.log((p[v]-q2[v])**2+(p[w]-q2[w])**2)+q2[v]
    elif p == q2:       #other corner
        Uinta = 1/2*(p[v]-q1[v])*math.log((p[v]-q1[v])**2+(p[w]-q1[w])**2)+q1[v]
        Uintb = 0
    else:            #normal integration over straight line segment
        Uinta = 1/2*(p[v]-q1[v])*math.log((p[v]-q1[v])**2+(p[w]-q1[w])**2)+q1[v]
        Uintb = 1/2*(p[v]-q2[v])*math.log((p[v]-q2[v])**2+(p[w]-q2[w])**2)+q2[v]
    
    if p[w] != q1[w]:
        Uinta = Uinta + (p[w]-q1[w])*math.atan((p[v]-q1[v])/(p[w]-q1[w]))    #when p and q dont lie on the same line
    if p[w] != q2[w]:
        Uintb = Uintb + (p[w]-q2[w])*math.atan((p[v]-q2[v])/(p[w]-q2[w]))
    Uint = 1/(2*pi)*(Uintb - Uinta)
    return Uint

def intDgreens(p,q1,q2):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                   # v is variable to integrate over
    w = (1 if q1[1]==q2[1] else 0)                  # w is other varaible, held constant over segment 
    if p[w] == q1[w]:   #p and q on same border
        if round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2):
            intDgreens.counter += 1
            if q2[v] > q1[v]:
                return -1/2   # self terms
            else:
                return 1/2
        else:
            return 0      #border terms
    else:
        DUinta = -1/(2*pi)*math.atan((p[v]-q1[v])/(p[w]-q1[w]))
        DUintb = -1/(2*pi)*math.atan((p[v]-q2[v])/(p[w]-q2[w]))
    DUint = DUintb - DUinta
    return DUint
#=============================================================================
    # MAIN CODE #
#=============================================================================
intDgreens.counter = 0    

# Create environment
#Ob1 = np.loadtxt(fname = 'node_coord_box.txt',delimiter = ",") #coordinates of obstacle 1
ob1 = [] #coordinates of obstacle 
bounds = [] #coordinates of outer domain, (environment is 1X2 in dimension). These are also the coordinates of all
#out points p. All are spaced 0.1 apart. additionally, they are ordered ( bounds[i]-bounds[i+1]=0.1)
x = 0
y = 0
for i in range(20):
    bounds.append([round(x,2),round(y,2)])          #must round because imperfect adding of decimals
    x += 0.1
for i in range(10):
    bounds.append([round(x,2),round(y,2)])
    y += 0.1
for i in range(20):
    bounds.append([round(x,2),round(y,2)])
    x -= 0.1
for i in range(10):
    bounds.append([round(x,2),round(y,2)])
    y -= 0.1    
boundsA = np.array(bounds)
start = [0.2, 0.2] #start coordinates
startA = np.array(start)
start_u = 1 #start potential
goal = [1.75, 0.8] #goal coordinates
goalA = np.array(goal)
goal_u = 0  #goal potential

# Plot environment
plt.figure(1)
plt.plot(boundsA[:,0],boundsA[:,1], 'ro')
plt.plot(start[0],start[1], 'ro')
plt.plot(goal[0],goal[1], 'ro')
#plt.plot(ob1[:,0],ob1[:,1], 'ro')

# Set up integral equations for empty boundary
C_Q = []  #smoothness factor for points p
gam_Q = [] #known and unknown derivative at boundary point p
u_Q  = [] #known and unknown potential at boundary point p
seg = [] #M outer box segment coordinates 
bounds_sg = [] #points of start and goal
for i in bounds:
    if (i == [0,1]) or (i==[2,1]) or (i==[0,0]) or (i==[2,0]):
        C_Q.append(1/4) #corners
    else:
        C_Q.append(1/2)
for i in range(0, len(bounds)):
        seg.append(list_avg(bounds[i-1],bounds[i]))
#seg.append(start)
#seg.append(goal)  %handle separartely
C_Q.append(1) #start position contained entirely in space
C_Q.append(1) #end position contained entirely in space
# for Behavior 4 of paper, we specify gamma (normal derivative) on all outer boundaries and solve for u (the potential).
# At the start and end points we specify the potential and calculate the normal derivatives.
for i in bounds:
    gam_Q.append(0)
gam_Q.append(1) #for start point, unknown
gam_Q.append(1) #for goal point, unknown
for i in bounds:
    u_Q.append(1)
u_Q.append(start_u) #for start point
u_Q.append(goal_u) #for goal point

#set up matrix A (unknowns)
lseg = 0.1 # length of all outer box segments
lseg_sg = 0.1 #length of start/goal segments
bounds.append(start); bounds.append(goal);
M = len(bounds)
A = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        if j >= M-2:               #start and goal positions where potential is known
            A[i,j] = -intgreens(bounds[i], list(np.subtract(bounds[j],[0.05, 0])), list(np.add(bounds[j],[0.05, 0]))) #make little segments for start and goal
        else:
            if j == M-3:                #because of indexing
                A[i,j] = intDgreens(bounds[i], seg[j], seg[0])
                if i == j:
                    A[i,j] += C_Q[j]
            elif C_Q[j] == 1/4:           #corner segments
                A[i,j] = intDgreens(bounds[i], seg[j], bounds[j])      
                A[i,j] += intDgreens(bounds[i], bounds[j], seg[j+1])  
                if i == j:
                    A[i,j] += 1/2 ############C_Q[j]  
                    A[i,j] += /2 #for self terms
            else:
                A[i,j] = intDgreens(bounds[i], seg[j], seg[j+1]) 
                if i == j:
                    A[i,j] += C_Q[j]
       # elif i == j:
       #     A[i,j] = C_Q[j] + lseg  
       # else:
       #     A[i,j] = lseg
                
#set up matrix b (knowns)
b = np.zeros(M)
for i in range(M):
    for j in range(M):
        if j<M-2:
            if j == M-3:                #because of indexing
                b[i] += gam_Q[j]*intgreens(bounds[i], seg[j], seg[0])  
            elif C_Q[j] == 1/4:           #corner segments
                b[i] += gam_Q[j]*intgreens(bounds[i], seg[j], bounds[j])      
                b[i] += gam_Q[j]*intgreens(bounds[i], bounds[j], seg[j+1])  
            else:
                b[i] += gam_Q[j]*intgreens(bounds[i], seg[j], seg[j+1])  
        else:
            if i == j:                                     
                b[i] += -(C_Q[j]+intDgreens(bounds[i], list(np.subtract(bounds[j],[0.05, 0])), list(np.add(bounds[j],[0.05, 0]))))*u_Q[j]     ############            
            else:
                b[i] += -intDgreens(bounds[i], list(np.subtract(bounds[j],[0.05, 0])), list(np.add(bounds[j],[0.05, 0])))*u_Q[j]
print('Im here')               
print(gam_Q)


#solve matrix equation and place unknown U and gam in their respective arrays. 
coef = np.linalg.solve(A,b)
for i in range(M):                #add the unkknown gam and u values to their respective arrays
    if i < M-2:
        u_Q[i] = coef[i]
    else:
        gam_Q[i] = coef[i]
        
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(list(boundsA[:,0]), list(boundsA[:,1]), u_Q[0:M-2], color='blue')
plt.show()

#use this code to check that the potential at each boundary point is correct
# =============================================================================
# U = np.zeros(M)                    #find potential at an arbitary point
# for i in range(M):
#     for j in range(M):
#         if j < M-2:     
#             if j == M-3:                  #because of indexing
#                 U[i] += gam_Q[j]*intgreens(bounds[i], seg[j], seg[0])  
#             elif C_Q[j] == 1/4:           #corner segments
#                 U[i] += gam_Q[j]*intgreens(bounds[i], seg[j], bounds[j])      
#                 U[i] += gam_Q[j]*intgreens(bounds[i], bounds[j], seg[j+1])  
#             else:
#                 U[i] += gam_Q[j]*intgreens(bounds[i], seg[j], seg[j+1]) 
#             U[i] += -lseg_sg*u_Q[j]
#         else: 
#             U[i]+= gam_Q[i]*intgreens(bounds[i], list(np.subtract(bounds[j],[0.05, 0])), list(np.add(bounds[j],[0.05, 0])))
#             U[i]+= -u_Q[j]*lseg_sg
#     U[i] *= 1/C_Q[i]
# =============================================================================
    
   

# ====================================================================================================
# # Plot trajectory
# fig, ax=plt.subplots()
# for x,y in path:                 #plot path
#     ax.plot(x,y, marker="o", color="blue", markeredgecolor="black") 
# for i in bounds:                 #plot boundaries
#     ax.plot(nodes[i,0],nodes[i,1], marker='X')
# xdata, ydata= [],[]
# line, = plt.plot( [], [], lw=3 )
# ====================================================================================================
