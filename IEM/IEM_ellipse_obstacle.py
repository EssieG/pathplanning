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
from meshEllipse import *
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

def ellipse_collocation(N, center_x1 = 1 , semi_major_axis_scale = 1):
    '''Default center is [1, 1/2], with a major-to-minor axis length ratio of 2:1. 
    x1 is the major axis, x2 is the minor axis.'''
    t = np.linspace(0, 2, N)
    x1 = np.cos(pi*t)*semi_major_axis_scale + center_x1
    x2 = (1/2*(np.sin(pi*t)))*semi_major_axis_scale +center_x1/2 
    x = np.column_stack((x1,x2))
    return x, t

def outside_ellipse(pos, c, r):
    '''pos is list of positions Xx2. returns Xx2 array of boolean True/False if inside ellipse'''
    inequality = (pos[:,0]-c[0])**2/r[0]**2 + (pos[:,1]-c[1])**2/r[1]**2
    valid = pos[inequality > 1]
    return valid

def ellipse_knots(N, center_x1 = 1 , semi_major_axis_scale = 1):
    t = np.linspace(0, 2, N)
    tcoll = (t[:-1]+t[1:])/2
    x1 = np.cos(pi*tcoll)*semi_major_axis_scale + center_x1
    x2 = (1/2*(np.sin(pi*tcoll)))*semi_major_axis_scale + center_x1/2 
    x = np.column_stack((x1,x2))
    return x

def dist(x1,x2):
    '''Distance between two points'''
    r = math.sqrt((x2[0]-x1[0])^2+(x2[1]-x1[1])^2)
    return r

def get_length(t1, t2, smas, c):
    integral = lambda t : sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    arclength = scipy.integrate.quad(integral, t1, t2)
    return arclength[0]

def plot_greens(p, t1, t2, smas, c):
    Greens = lambda t : -1/(2*pi) * ln(sqrt((p[0]-(np.cos(pi*t)*smas + c))**2+(p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2)) * sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    points = int(1e4) #Number of points
    xlist = np.linspace(t1,t2,points)
    ylist = list(map(Greens, xlist))
    fig = plt.figure()
    #plt.ylim([0,1])
    plt.plot(xlist, ylist)
    
def plot_Dgreens(p, t1, t2, smas, c, q1, q2):
    exterior_normal = find_normal(q1,q2)
    DGreens = lambda t : 1/(2*pi)/((p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2+(p[0]-(np.cos(pi*t)*smas + c))**2)*((p[0]-(np.cos(pi*t)*smas + c))*exterior_normal[0] + (p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))*exterior_normal[1]) * sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    points = int(1e4) #Number of points
    xlist = np.linspace(t1,t2,points)
    ylist = list(map(DGreens, xlist))
    fig = plt.figure()
    #plt.ylim([0.4,.6])
    plt.plot(xlist, ylist)

def intgreens(p, t1, t2, smas, c, is_selfterm = False):
    '''Input is reference point p and the segment points to integrate Greens function over. Output is 
    integral of Green's potential at point p'''
    #x_t = lambda t1 :  (np.cos(pi*t)*smas + c)
    #y_t = lambda t2 : (1/2*np.sin(pi*t)*smas + c/2) 
    Greens = lambda t : -1/(2*pi) * ln(sqrt((p[0]-(np.cos(pi*t)*smas + c))**2+(p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2)) * sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    if is_selfterm:
        t_mid = (t1+t2)/2
        Gint = scipy.integrate.quad(Greens, t1, t_mid)[0] + scipy.integrate.quad(Greens, t_mid, t2)[0]
    else:
        Gint = scipy.integrate.quad(Greens, t1, t2)[0]
    return Gint

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

def intDgreens(p, t1, t2, smas, c, q1, q2, is_selfterm = False):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    exterior_normal = find_normal(q1,q2)
    DGreens = lambda t : 1/(2*pi)/((p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))**2+(p[0]-(np.cos(pi*t)*smas + c))**2)*((p[0]-(np.cos(pi*t)*smas + c))*exterior_normal[0] + (p[1]-((1/2*(np.sin(pi*t)))*smas + c/2))*exterior_normal[1]) * sqrt((-pi*np.sin(pi*t)*smas)**2+ (pi/2*np.cos(pi*t)*smas)**2)
    if is_selfterm:
        t_mid = (t1+t2)/2
        DGint = scipy.integrate.quad(DGreens, t1, t_mid)[0] + scipy.integrate.quad(DGreens, t_mid, t2)[0]
    else:
        DGint = scipy.integrate.quad(DGreens, t1, t2)[0] 
    return DGint


#============================================================================
    # MAIN CODE #
#============================================================================
# Set Parameters   
N_environment = 30                 #SPECIFIC number of points on y (odd number)
N_obstacle = 20
N_total = N_environment + N_obstacle
center_of_environment_x = 1        #Where is the center x coordinate of the ellipse. center y will be 1/2*center_x
ratio_of_environment = 2*1           #scale of environment wrt environment, "smas"
center_of_obstacle_x = 1    
ratio_of_obstacle = .5

# Discretize outer boundary
environment, t_environment = ellipse_collocation(N_environment+1, center_of_environment_x, ratio_of_environment)  #(N_environment+1) discretized bounds of points,p, with corresponding parameter values t
t_p_environment = np.absolute(t_environment[:-1] + t_environment[1:])/2  #(N_environment) t for points p. t[0]<p_t[0]<t[1]
p_environment = ellipse_knots(N_environment+1,center_of_environment_x, ratio_of_environment) #(N_environment)
obstacle, t_obstacle = ellipse_collocation(N_obstacle+1, center_of_obstacle_x, ratio_of_obstacle)
t_p_obstacle = np.absolute(t_obstacle[:-1] + t_obstacle[1:])/2  
p_obstacle = ellipse_knots(N_obstacle+1,center_of_obstacle_x, ratio_of_obstacle) 
p_all = np.vstack((p_environment, p_obstacle))

# Set up boundary conditions for collocation
c_Q = np.full(N_total,1/2)   # 1/2 for environment
u_Q = np.ones(N_total)       #obstacles boundary set to 1
u_Q[0:N_environment] = 0     #set outer environement equal to zero 


# Set up matrix A (unknowns)         #ipdb.set_trace()
A = np.zeros((N_total,N_total))
for i in range(N_total):             #sum over boundary elements and obstacle elements
    for k in range(N_environment): 
        is_self = True if i == k else False
        A[i,k] = intgreens(p_all[i], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x, is_self)
    for l in range(N_obstacle):
        is_self = True if i == N_environment+l else False
        A[i,N_environment+l] = - intgreens(p_all[i], t_obstacle[l], t_obstacle[l+1], ratio_of_obstacle, center_of_obstacle_x, is_self)

# Set up matrix b (knowns)
b = np.zeros(N_total)
for i in range(N_total):
    for k in range(N_environment):
        is_self = True if i == k else False
        b[i] +=  u_Q[k]*intDgreens(p_all[i], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x, environment[k], environment[k+1], is_self)
        if is_self:
            b[i] +=  u_Q[i]*c_Q[i]
    for l in range(N_obstacle):
        is_self = True if i == N_environment+l else False
        b[i] +=  - u_Q[N_environment+l]*intDgreens(p_all[i], t_obstacle[l], t_obstacle[l+1], ratio_of_obstacle, center_of_obstacle_x, obstacle[l], obstacle[l+1], is_self)
        if is_self:
            b[i] += u_Q[i]*c_Q[i]

            
# Solve matrix equation for gam on environment and obstacle
gam_Q = la.solve(A,b) 
#gam_Q = la.lstsq(A,b) 
#gam_Q=gam_Q[0]  
#gam_Q[u_indices] = coeff[u_indices]
#u_Q[gam_indices] = coeff[gam_indices]         

#============================================================================
    # DEBUGGING #
#============================================================================

# Solve for other things
#len_ellipse = get_length(t_environment[0], t_environment[-1], ratio_of_environment, center_of_environment_x)  #cirumference of ellipse, should be 4.84422
#int_D_greens_around_boundary = 0 #integrate Dgreens around a single arbitrary interior point. Should equal one
#int_greens_around_boundary = 0 #for checking the sign
#for k in range(N_environment): 
#    int_D_greens_around_boundary += intDgreens([1.75,0.4], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x, environment[k], environment[k+1])
#    int_greens_around_boundary += intgreens([1.75,0.4], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x)

#sigma = 0  #for exterior problem
#for k in range(N_environment):  
#    sigma += gam_Q[k]*get_length(t_environment[k-1], t_environment[k], ratio_of_environment, center_of_environment_x)

#temp = intDgreens(p_environment[0], t_environment[0], t_environment[1], ratio_of_environment, center_of_environment_x, environment[0], environment[1],True)
#temp1 = plot_Dgreens(p_environment[0], t_environment[0], t_environment[1], ratio_of_environment, center_of_environment_x, environment[0], environment[1])
#temp = intgreens(p_environment[1], t_environment[0], t_environment[1], ratio_of_environment, center_of_environment_x)
#temp1 = plot_greens(p_environment[1], t_environment[0], t_environment[1], ratio_of_environment, center_of_environment_x)

#===========================================================================
    # Generate Plots #
#===========================================================================

# Plot collocation boundary & normals to verify
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(environment[:,0], environment[:,1], 'ro')
plt.plot(p_environment[:,0], p_environment[:,1], 'bo')
plt.plot(obstacle[:,0], obstacle[:,1], 'go')
plt.plot(p_obstacle[:,0], p_obstacle[:,1], 'bo')
plt.axis('equal')
normals1 = np.zeros((N_environment,2))
normals2 = np.zeros((N_obstacle,2))
for i in range(N_environment):
    normals1[i] = find_normal(environment[i], environment[i+1])
for i in range(N_obstacle):
    normals2[i] = find_normal(obstacle[i], obstacle[i+1])
ax.quiver(p_environment[:,0],p_environment[:,1], normals1[:,0], normals1[:,1], color ='r') 
ax.quiver(p_obstacle[:,0],p_obstacle[:,1], normals2[:,0], normals2[:,1], color ='r')#U and V are the x and y components of the normal vectors


# Plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients (NOT WRITTEN)
fig = plt.figure()
ax = Axes3D(fig)
u_bound_est = [] 
for i in range(N_total):
    temp = 0
    for k in range(N_environment):
        is_self = True if i == k else False
        temp += 2*(gam_Q[k]*intgreens(p_all[i], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x, is_self) - u_Q[k]*intDgreens(p_all[i], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x, environment[k], environment[k+1], is_self))
    for l in range(N_obstacle):
        is_self = True if i == N_environment+l else False
        temp += -2*(gam_Q[N_environment+l]*intgreens(p_all[i], t_obstacle[l], t_obstacle[l+1], ratio_of_obstacle, center_of_obstacle_x, is_self) - u_Q[N_environment+l]*intDgreens(p_all[i], t_obstacle[l], t_obstacle[l+1], ratio_of_obstacle, center_of_obstacle_x, obstacle[l], obstacle[l+1], is_self))
    u_bound_est.append(temp)
ax.plot(list(p_all[:,0]), list(p_all[:,1]), u_bound_est, color='blue')
plt.plot(environment[:,0], environment[:,1], 'ro')


# Solve for points inside the domain and plot potential inside domain
n = 10; r = np.array([0.500001,0.250001]); c = [1,.5]; ng = ellipse_grid_count(n, r, c); filename = 'myellipse2.png' 
xy_inner = ellipse_grid_points(n, r, c, ng).T
xy_outer = ellipse_grid_points(2*n, 4*r, c, ellipse_grid_count(2*n, 4*r, c)).T
xy = outside_ellipse(xy_outer, c, r)
#ellipse_grid_display( 2*n+1, 2*r, c, ellipse_grid_count(2*n+1, 2*r, c) , xy_outer, filename )

phi = np.zeros((len(xy),1))
for i in range(len(xy)):                           
    temp=0
    for k in range(N_environment):
        temp += (gam_Q[k]*intgreens(xy[i], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x) - u_Q[k]*intDgreens(xy[i], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x, environment[k], environment[k+1]))
    for l in range(N_obstacle):
        temp += -(gam_Q[N_environment+l]*intgreens(xy[i], t_obstacle[l], t_obstacle[l+1], ratio_of_obstacle, center_of_obstacle_x) - u_Q[N_environment+l]*intDgreens(xy[i], t_obstacle[l], t_obstacle[l+1], ratio_of_obstacle, center_of_obstacle_x, obstacle[l], obstacle[l+1]))
    phi[i] = temp    
plot3D(xy[:,0],xy[:,1], phi[:,0]) 


#condition number of matrix
#la.cond(A)


