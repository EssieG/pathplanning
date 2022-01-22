"""
Script: IEM_ellipse.py
Author: Esther Grossman
Date: 3/29/21

Solving with pulse basis functions. The start and goal points are ON the boundary.
Potential specified EVERYWHERE. Normal derivative calculated on all boundaries
and start/goal position.

"""

import ipdb
import numpy as np
import numpy.linalg as la
import math
pi = math.pi
sqrt = np.lib.scimath.sqrt
ln = np.lib.scimath.log
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

def plot3D(x,y,z):
    '''Plots anything in 3D. Intended for showing potential surface over 2D plane.'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title('IEM grid')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi$')
    plt.show()

def make_ellipse(N, center_x1 = 1 , semi_major_axis_scale = 1):
    '''Default center is [1, 1/2], with a major-to-minor axis length ratio of 2:1. 
    x1 is the major axis, x2 is the minor axis.'''
    t = np.linspace(0, 2, N)
    x1 = np.cos(pi*t)*semi_major_axis_scale + center_x1
    x2 = (1/2*(np.sin(pi*t)))*semi_major_axis_scale +center_x1/2 
    x = np.column_stack((x1,x2))
    return x, t

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

def intgreens(p, t1, t2, smas, c):
    '''Input is reference point p and the segment points to integrate Greens function over. Output is 
    integral of Green's potential at point p'''
    #x_t = lambda t1 :  (np.cos(pi*t1)*smas + c)
    #y_t = lambda t2 : ((1/2*(np.sin(pi*t2)))*smas + c/2) 
    Greens = lambda t : -1/(2*pi) * ln(sqrt((p[0]-(np.cos(pi*t)*smas + c))**2+(p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2)) * sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    Gint = scipy.integrate.quad(Greens, t1, t2)
    return Gint[0]
    
# =============================================================================
#     if p[w] == q1[w] and round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms
#         func = lambda x : -1/(pi)*np.lib.scimath.log(np.lib.scimath.sqrt((p[v]-x)**2+(p[w]-q1[w])**2))
#         Uint = scipy.integrate.quad(func, p[v], q2[v])
#     else:  
#         func = lambda x : -1/(2*pi)*np.lib.scimath.log(np.lib.scimath.sqrt((p[v]-x)**2+(p[w]-q1[w])**2))
#         Uint = scipy.integrate.quad(func, q1[v], q2[v])
#     Uint = Uint[0]
#     return Uint
# =============================================================================

def find_normal(q1,q2):
    if np.round(q2[1], 3) == np.round(q1[1],3):
            normal = np.array([0,1])
    else:
        m = -(q2[0]-q1[0])/(q2[1]-q1[1])
        normal_temp = np.array([1, m])
        normal = normal_temp/np.linalg.norm(normal_temp)
    if q2[1] < q1[1]:
        normal *= -1
    return normal

def intDgreens(p, t1, t2, smas, c, q1, q2):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    exterior_normal = find_normal(q1,q2) 
    DGreens = lambda t : 1/(2*pi)*((p[0]-(np.cos(pi*t)*smas + c))/((p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2+(p[0]-(np.cos(pi*t)*smas + c))**2)*exterior_normal[0] + (p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))/((p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2+(p[0]-(np.cos(pi*t)*smas + c))**2)*exterior_normal[1]) * sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    DGint = scipy.integrate.quad(DGreens, t1, t2) 
    return DGint[0]
# =============================================================================
#     Q = np.subtract(q2,q1)
#     m = np.zeros(2); m[w]=1;  #positive normal vector
#     nw = (-1 if np.cross(Q,m) > 0 else 1)  #outer boundary normal pointing outward from walls 
#     norm_out = (-1 if ext == False else 1) #if negative, normal is pointing inward, such as for obstacles 
#         
#     if p[w] == q1[w]: 
#         if round(abs(p[v]-q1[v]),2) == round(abs(p[v]-q2[v]),2): #self-terms solution
#             if q2[v]>q1[v]:
#                 return -1/2
#             else:
#                 return 1/2
#     dfunc = lambda x : (p[w]-q1[w])/(2*pi*((p[w]-q1[w])**2+(p[v]-x)**2))*nw*norm_out  #gradient of greens function with respect to w (normal direction)
#     DUint = scipy.integrate.quad(dfunc, q1[v], q2[v]) 
#     DUint = DUint[0]
# =============================================================================

def intDgreensA(p,q1,q2):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of Green's potential at point p'''
    v = (0 if q1[1]==q2[1] else 1)                  # v is variable to integrate over
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
N_obstacle = 11     #SPECIFIC number of points on x (check that there are no vertical normals)
N_environment = 15     #SPECIFIC number of points on y (odd number)
N_total = N_obstacle + N_environment        #total number of discretized points
center_of_environment_x = 1
ratio_of_environment = 1           #size of environment wrt environment
ratio_of_obstacle = 1/2
center_of_obstacle_x = 1
#ipdb.set_trace()

# Discretize outer boundary
obstacle_temp, t_obstacle_temp = make_ellipse(N_obstacle+1, center_of_obstacle_x, ratio_of_obstacle) #make obstcle half as big, centered, discretized obstacle
environment_temp, t_environment_temp = make_ellipse(N_environment+1)  #discretized outer domain, redundant 1st point
p_obstacle = np.absolute(obstacle_temp[:-1] + obstacle_temp[1:])/2
p_environment = np.absolute(environment_temp[:-1] + environment_temp[1:])/2
obstacle, environment = obstacle_temp[1:], environment_temp[1:]
t_p_obstacle = np.absolute(t_obstacle_temp[:-1] + t_obstacle_temp[1:])/2           #t for points in obstacle
t_p_environment = np.absolute(t_environment_temp[:-1] + t_environment_temp[1:])/2  #t for points in environment
t_obstacle, t_environment = t_obstacle_temp[1:], t_environment_temp[1:]          #t for boundary segments
t_p_all = np.concatenate((t_p_environment, t_p_obstacle))
p_all = np.vstack((p_environment, p_obstacle))

# Plot collocation boundary
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(environment[:,0], environment[:,1], 'ro')
plt.plot(p_environment[:,0], p_environment[:,1], 'bo')
plt.plot(p_obstacle[:,0], p_obstacle[:,1], 'bo')
plt.plot(obstacle[:,0], obstacle[:,1], 'ro')
plt.axis('equal')

# Set up boundary conditions for collocation
c_Q = np.full(N_total,1/2) 
u_Q = np.ones(N_total)       #known and unknown potential at boundary point p, obstacles and start set to 0
u_Q[:N_environment] = 0      #set outer environement equal to zero
gam_Q = np.ones(N_total)     #unknow derivatives


# Set up matrix A (unknowns) 
A = np.zeros((N_total,N_total))
for i in range(N_total):       #sum over boundary elements and obstacle elements separately for indexing purposes
    for k in range(N_environment):           
        A[i,k] = intgreens(p_all[i], t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x)
    for k in range(N_obstacle):         
        A[i,N_environment+k] = intgreens(p_all[i], t_obstacle[k-1], t_obstacle[k], ratio_of_obstacle, center_of_obstacle_x)

#Plot normals to check direction
normals1 = np.zeros((N_environment,2))
normals2 = np.zeros((N_obstacle,2))
for i in range(N_environment):
    normals1[i] = find_normal(environment[i-1], environment[i])
for i in range(N_obstacle):
    normals2[i] = find_normal(obstacle[i-1], obstacle[i])
ax.quiver(p_obstacle[:,0],p_obstacle[:,1], normals2[:,0], normals2[:,1], color ='r') 
ax.quiver(p_environment[:,0],p_environment[:,1], normals1[:,0], normals1[:,1], color ='r') #U and V are the x and y components of the normal vecots

# Set up matrix b (knowns)
b = np.zeros(N_total)
for i in range(N_total):
    for k in range(N_environment):    
        b[i] += u_Q[k]*intDgreens(p_all[i], t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x, environment[k-1], environment[k])
        if i == k:
            b[i] += u_Q[i]*c_Q[i]
    for k in range(N_obstacle):
        b[i] += u_Q[N_environment+k]*intDgreens(p_all[i], t_obstacle[k-1], t_obstacle[k], ratio_of_obstacle, center_of_obstacle_x, obstacle[k-1], obstacle[k])
        if i == N_environment+k:
            b[i] += u_Q[i]*c_Q[i]

# Solve matrix equation 
gam_Q = la.solve(A,b)             #coeff are all the values of gam


#===========================================================================
    # Generate Plots #
#===========================================================================

# plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients
all_bounds = np.vstack((environment, obstacle))
fig = plt.figure()
ax = Axes3D(fig)
pA = np.array(p_all)
u_bound_est = [] 
for i in range(N_total):
    temp = 0
    for k in range(N_environment):
        temp += 2*(gam_Q[k]*intgreens(p_all[i], t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x)-u_Q[k]*intDgreens(p_all[i], t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x, environment[k-1], environment[k]))
    for k in range(N_obstacle):
        temp += 2*(gam_Q[N_environment+k]*intgreens(p_all[i], t_obstacle[k-1], t_obstacle[k], ratio_of_obstacle, center_of_obstacle_x)-u_Q[N_environment+k]*intDgreens(p_all[i], t_obstacle[k-1], t_obstacle[k], ratio_of_obstacle, center_of_obstacle_x, obstacle[k-1], obstacle[k]))
    u_bound_est.append(temp)
ax.plot(list(pA[:,0]), list(pA[:,1]), u_bound_est, color='blue')
plt.plot(all_bounds[:,0], all_bounds[:,1], 'ro')

# =============================================================================
# # Plot initial potential on the boundary        
# fig = plt.figure()
# ax = Axes3D(fig)
# pA = np.array(p[0:M])
# ax.plot(list(pA[:,0]), list(pA[:,1]), u_Q[0:M], color='blue')
# plt.plot(boundsA[:,0],boundsA[:,1], 'ro')
# =============================================================================

# Solve for points inside the domain and plot potential inside domain
X,Y = np.mgrid[0.15:2-0.15:30j, 0.15:1-0.15:30j]
valid1 = np.logical_or(X<1.9 , X>c1+3*lseg)
valid2 = np.logical_or(Y<c2 , Y>c2+3*lseg)
valid = np.logical_or(valid1, valid2)
#Xvalid, Yvalid = X[valid], Y[valid]
xy = np.vstack((X.flatten(), Y.flatten())).T #coordinates of interior as MNx2 vector
X[~valid] = np.nan; Y[~valid] = np.nan
xybool = np.vstack((X.flatten(), Y.flatten())).T

phi = np.zeros((len(xy),1))
for i in range(len(xy)):
    if np.isnan(xybool[i]).any():
        phi[i] = 1; 
        continue
    for k in range(N_environment):
        temp += 2*(gam_Q[k]*intgreens(xy[i], t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x)-u_Q[k]*intDgreens(p_all[i], t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x, environment[k-1], environment[k]))
    for k in range(N_obstacle):
        temp += 2*(gam_Q[N_environment+k]*intgreens(xy[i], t_obstacle[k-1], t_obstacle[k], ratio_of_obstacle, center_of_obstacle_x)-u_Q[N_environment+k]*intDgreens(p_all[i], t_obstacle[k-1], t_obstacle[k], ratio_of_obstacle, center_of_obstacle_x, obstacle[k-1], obstacle[k]))

plot3D(xy[:,0],xy[:,1], phi[:,0])       
     


#condition number of matrix
#la.cond(A)


