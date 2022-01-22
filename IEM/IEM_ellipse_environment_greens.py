"""
Script: IEM_ellipse.py
Author: Esther Grossman
Date: 7/1/21

Solving with pulse basis functions for the greens potential at interior points to the enivronment
using purely Neumann boundary conditions.

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
    ax.set_zlabel('$g$')
    plt.show()

def ellipse_collocation(N, center, radii):
    '''Default center is [1, 1/2], with a major-to-minor axis length ratio of 2:1. 
    x1 is the major axis, x2 is the minor axis.'''
    t = np.linspace(0, 2, N)
    x1 = np.cos(pi*t)*radii[0] + center[0]
    x2 = np.sin(pi*t)*radii[1] + center[1] 
    x = np.column_stack((x1,x2))
    return x, t

def ellipse_knots(N, center, radii):
    t = np.linspace(0, 2, N)
    tcoll = (t[:-1]+t[1:])/2
    x1 = np.cos(pi*tcoll)*radii[0] + center[0]
    x2 = np.sin(pi*tcoll)*radii[1] + center[1]  
    x = np.column_stack((x1,x2))
    return x

def outside_ellipse(pos, c, r):
    '''pos is list of positions Xx2. returns Xx2 array of boolean True/False if inside ellipse'''
    inequality = (pos[:,0]-c[0])**2/r[0]**2 + (pos[:,1]-c[1])**2/r[1]**2
    valid = pos[inequality > 1]
    return valid

def dist(x1,x2):
    '''Distance between two points'''
    r = math.sqrt((x2[0]-x1[0])^2+(x2[1]-x1[1])^2)
    return r

def get_length(t1, t2, center, radii):
    integral = lambda t : sqrt((-pi*np.sin(pi*t)*radii[0])**2 + (pi*np.cos(pi*t)*radii[1])**2)
    arclength = scipy.integrate.quad(integral, t1, t2)
    return arclength[0]

def plot_greens(p, t1, t2, center, radii):
    Greens = lambda t : -1/(2*pi) * ln(sqrt((p[0]-(np.cos(pi*t)*radii[0] + center[0]))**2+(p[1]-(np.sin(pi*t)*radii[1] + center[1]))**2)) * sqrt((-pi*np.sin(pi*t)*radii[0])**2+ (pi*np.cos(pi*t)*radii[1])**2)
    points = int(1e4) #Number of points
    xlist = np.linspace(t1,t2,points)
    ylist = list(map(Greens, xlist))
    fig = plt.figure()
    #plt.ylim([0,1])
    plt.plot(xlist, ylist)
    
def Dgreens(q, impulse_point, q1, q2):
    ''' evaluate the greens derivative at the impulse point with respect to the normal of point q.'''
    exterior_normal = find_normal(q1,q2)
    DGreens = 1/(2*pi)/((impulse_point[1]-q[1])**2+(impulse_point[0]-q[0])**2)*((impulse_point[0]-q[0])*exterior_normal[0] + (impulse_point[1]-q[1])*exterior_normal[1])
    return DGreens

def greens(q, impulse_point):
    ''' evaluate the greens derivative at the impulse point with respect to the normal of point q.'''
    Greens = -1/(2*pi) * ln(sqrt((q[0]-impulse_point[0])**2+(q[1]-impulse_point[1])**2)) 
    return Greens

def plot_Dgreens(p, t1, t2, center, radii, q1, q2):
    exterior_normal = find_normal(q1,q2)
    DGreens = lambda t : 1/(2*pi)/((p[1]-(np.sin(pi*t)*radii[1] + center[1]))**2+(p[0]-(np.cos(pi*t)*radii[0] + center[0]))**2)*((p[0]-(np.cos(pi*t)*radii[0] + center[0]))*exterior_normal[0] + (p[1]-(np.sin(pi*t)*radii[1] + center[1]))*exterior_normal[1]) * sqrt((-pi*np.sin(pi*t)*radii[0])**2+ (pi*np.cos(pi*t)*radii[1])**2)
    points = int(1e4) #Number of points
    xlist = np.linspace(t1,t2,points)
    ylist = list(map(DGreens, xlist))
    fig = plt.figure()
    #plt.ylim([0.4,.6])
    plt.plot(xlist, ylist)

def intgreens(p, t1, t2, center, radii, is_selfterm = False):
    '''Input is reference point p and the segment points to integrate Greens function over. Output is 
    integral of Green's potential at point p'''
    Greens = lambda t : -1/(2*pi) * ln(sqrt((p[0]-(np.cos(pi*t)*radii[0] + center[0]))**2+(p[1]-(np.sin(pi*t)*radii[1] + center[1]))**2)) * sqrt((-pi*np.sin(pi*t)*radii[0])**2+ (pi*np.cos(pi*t)*radii[1])**2)
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

def intDgreens(p, t1, t2, center, radii, q1, q2, is_selfterm = False):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p'''
    exterior_normal = find_normal(q1,q2)
    DGreens = lambda t : 1/(2*pi)/((p[1]-(np.sin(pi*t)*radii[1] + center[1]))**2+(p[0]-(np.cos(pi*t)*radii[0] + center[0]))**2)*((p[0]-(np.cos(pi*t)*radii[0] + center[0]))*exterior_normal[0] + (p[1]-(np.sin(pi*t)*radii[1] + center[1]))*exterior_normal[1]) * sqrt((-pi*np.sin(pi*t)*radii[0])**2+ (pi*np.cos(pi*t)*radii[1])**2)
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
N_environment = 40;                   #number on outer bundary and on each of start and goal
N_total = N_environment
center_environment = [1, 1]         #ellipse center
radii_environment = [1, 1]          #major and minor radius of ellipse
        

# Discretize outer boundary
environment, t_environment = ellipse_collocation(N_environment+1, center_environment, radii_environment)  #(N_environment+1) discretized bounds of points,p, with corresponding parameter values t
t_p_environment = np.absolute(t_environment[:-1] + t_environment[1:])/2  #(N_environment) t for points p. t[0]<p_t[0]<t[1]
p_environment = ellipse_knots(N_environment+1, center_environment, radii_environment) #(N_environment)
#Discretize space to get greens on interior points
#PUT ELLIPSE DISCRETIZATION HERE
#p_impulse =  [1,1]
p_impulse = [1.1,1.1]  #index 80
p_constant = center_environment

# Set up boundary conditions for collocation
c_Q = np.full(N_total,1/2) 
g_Q = np.ones(N_total)          #greens unknown everywhere on boundary
gam_Q = np.zeros(N_total)       #greens derivative zero on boundary
ellipse_circumference = get_length(t_environment[0], t_environment[-1], center_environment, radii_environment)


#ipdb.set_trace()
# Set up matrix A (unknowns) 
A = np.zeros((N_total,N_total))
for i in range(N_total):
    for k in range(N_environment):   #when integrating over environment segment
        is_self = True if i == k else False
        if i == N_total - 1:         #replace N-1 row with dirichlet node
            #A[i,k] = - intgreens(p_constant, t_environment[k], t_environment[k+1], center_environment, radii_environment)
            A[i,k] = - intDgreens(p_environment[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1], is_self)
            if is_self:
                A[i,k] += c_Q[i]
        else:
            A[i,k] = - intDgreens(p_environment[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1], is_self)
            if is_self:
                A[i,k] += c_Q[i]
    
        
# Set up matrix b (knowns)
b = np.zeros(N_total)
for i in range(N_total):
    b[i] = -1/ellipse_circumference + Dgreens(p_environment[i], p_impulse, environment[i], environment[i+1])
#b[N_total - 1] = greens(p_constant, p_impulse) - 1    #replace N-1 row with dirichlet node        
            
# Solve matrix equation 
g_Q = la.solve(A,b)    #get greens at every point
#gam_Q = la.lstsq(A,b)  
        

#============================================================================
    # DEBUGGING #
#============================================================================

# Solve for other things
#len_ellipse = get_length(t_environment[0], t_environment[-1], center_environment, radii_environment)  #cirumference of ellipse, should be 4.84422
#int_D_greens_around_boundary = 0 #integrate Dgreens around a single arbitrary interior point. Should equal one
#int_greens_around_boundary = 0 #for checking the sign
#for k in range(N_environment): 
#    int_D_greens_around_boundary += intDgreens([1.75,0.4], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1])
#    int_greens_around_boundary += intgreens([1.75,0.4], t_environment[k], t_environment[k+1], ratio_of_environment, center_of_environment_x)

#sigma = 0  #for exterior problem
#for k in range(N_environment):  
#    sigma += gam_Q[k]*get_length(t_environment[k-1], t_environment[k], center_environment, radii_environment)

#temp = intDgreens(p_environment[0], t_environment[0], t_environment[1], center_environment, radii_environment, environment[0], environment[1],True)
#temp1 = plot_Dgreens(p_environment[0], t_environment[0], t_environment[1], center_environment, radii_environment, environment[0], environment[1])
#temp = intgreens(p_environment[1], t_environment[0], t_environment[1], center_environment, radii_environment)
#temp1 = plot_greens(p_environment[0], t_environment[0], t_environment[1], center_environment, radii_environment)

#===========================================================================
    # Generate Plots #
#===========================================================================

# =============================================================================
# # Plot collocation boundary & normals to verify
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(environment[:,0], environment[:,1], 'ro')
# plt.plot(p_environment[:,0], p_environment[:,1], 'bo')
# plt.axis('equal')
# normals1 = np.zeros((N_environment,2))
# normals2 = np.zeros((N_startgoal,2))
# for i in range(N_environment):
#     normals1[i] = find_normal(environment[i], environment[i+1])
# for i in range(N_startgoal):
#     normals2[i] = find_normal(start[i], start[i+1])
# ax.quiver(p_environment[:,0],p_environment[:,1], normals1[:,0], normals1[:,1], color ='r') #U and V are the x and y components of the normal vectors
# ax.quiver(p_start[:,0],p_start[:,1], normals2[:,0], normals2[:,1], color ='r')
# ax.quiver(p_goal[:,0],p_goal[:,1], normals2[:,0], normals2[:,1], color ='r')
# =============================================================================

# =============================================================================
# # Plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients
# fig = plt.figure()
# ax = Axes3D(fig)
# u_bound_est = [] 
# for i in range(N_environment):
#     temp = 0
#     for k in range(N_environment):
#         is_self = True if i == k else False
#         temp += 2*(gam_Q[k]*intgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, is_self) - u_Q[k]*intDgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1], is_self))
#     #     
#     xy_environment = ellipse_grid_points(2*n, np.array(radii_environment)-[0.001,0.001], center_environment, ellipse_grid_count(2*n, np.array(radii_environment)-[0.001,0.001], center_environment)).T
#     u_bound_est.append(temp)
# ax.plot(list(p_environment[:,0]), list(p_environment[:,1]), u_bound_est, color='blue')
# plt.plot(environment[:,0], environment[:,1], 'ro')
# =============================================================================


# Solve for points inside the domain and plot potential inside domain
n = 5; r = np.array(radii_environment)-[0.001,0.001]; c = center_environment; ng = ellipse_grid_count(n, r, c); filename = 'myellipse.png' 
xy = ellipse_grid_points(n, r, c, ng).T

#ellipse_grid_display( n, r, c, ng, xy_horizontal, filename )
phi = np.zeros((len(xy),1))
for i in range(len(xy)):                           #There will not be self-terms on the interior
    temp=0
    if ( np.round(xy[i], 4) == np.round(p_impulse, 4) ).all():
        continue
    for k in range(N_environment):
        temp += g_Q[k]*intgreens(xy[i], t_environment[k], t_environment[k+1], center_environment, radii_environment)
    temp += greens(xy[i], p_impulse)
    phi[i] = temp
plot3D(xy[:,0],xy[:,1], phi[:,0])       
     
#check symmetry of greens
for i in range(len(phi)):
    print(i,xy[i], phi[i])

#condition number of matrix
#la.cond(A)
#la.det(A)  #check determinant


