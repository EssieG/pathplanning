#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:45:39 2022

Code that replicates the IEM algorithm 

@author: ubuntu
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
    ax = fig.add_subplot(projection='3d')
    #ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title('IEM grid')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi$')
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

def potential_at_point_in_obstacle_domain(point , u, gam, N_env, N_ob, t_env, c_env, r_env, env, t_ob, c_ob, r_ob, ob):
    potential=0
    for k in range(N_env):
        potential += (gam[k]*intgreens(point, t_env[k], t_env[k+1], c_env, r_env) - u[k]*intDgreens(point, t_env[k], t_env[k+1], c_env, r_env, env[k], env[k+1]))
    for l in range(N_ob):
        potential += -(gam[N_env+l]*intgreens(point, t_ob[l], t_ob[l+1], c_ob, r_ob) - u[N_env+l]*intDgreens(point, t_ob[l], t_ob[l+1], c_ob, r_ob, ob[l], ob[l+1]))
    return potential    

def potential_at_point_in_workspace(point , u, gam, N_env, N_sg, t_env, c_env, r_env, env, t_sg, c_s, c_g, r_sg, start, goal):
    potential=0
    for k in range(N_env):
        potential += (gam[k]*intgreens(point, t_env[k], t_env[k+1], c_env, r_env) - u[k]*intDgreens(point, t_env[k], t_env[k+1], c_env, r_env, env[k], env[k+1]))
    for k in range(N_startgoal):
        potential += - (gam[N_env+k]*intgreens(point, t_sg[k], t_sg[k+1], c_s, r_sg) - u[N_env+k]*intDgreens(point, t_sg[k], t_sg[k+1], c_s, r_sg, start[k], start[k+1]))
    for k in range(N_startgoal): 
        potential += - (gam[N_env+N_sg+k]*intgreens(point, t_sg[k], t_sg[k+1], c_g, r_sg) - u[N_env+N_sg+k]*intDgreens(point, t_sg[k], t_sg[k+1], c_g, r_sg, goal[k], goal[k+1]))
    return potential 
    

#%%============================================================================
    # MAIN CODE #
#============================================================================
# Set Parameters   
N_environment = 30; N_startgoal = 11;     #number on outer bundary and on each of start and goal, check normal directons are all right
N_total = N_environment + 2 * N_startgoal
center_environment = [1, 1]         #ellipse center
radii_environment = [1.5, 1.5]          #major and minor radius of ellipse
center_start = [.5, 0.5] 
center_goal = [1.5, 1.55] 
radii_startgoal = [0.05, 0.05]
        

# Discretize outer boundary
environment, t_environment = ellipse_collocation(N_environment+1, center_environment, radii_environment)  #(N_environment+1) discretized bounds of points,p, with corresponding parameter values t
t_p_environment = np.absolute(t_environment[:-1] + t_environment[1:])/2  #(N_environment) t for points p. t[0]<p_t[0]<t[1]
p_environment = ellipse_knots(N_environment+1, center_environment, radii_environment) #(N_environment)
start, t_startgoal = ellipse_collocation(N_startgoal+1, center_start, radii_startgoal)  
goal, __ = ellipse_collocation(N_startgoal+1, center_goal, radii_startgoal)  
t_p_startgoal = np.absolute(t_startgoal[:-1] + t_startgoal[1:])/2  
p_start = ellipse_knots(N_startgoal+1, center_start, radii_startgoal)
p_goal = ellipse_knots(N_startgoal+1, center_goal, radii_startgoal) 
p_all = np.vstack((p_environment, p_start, p_goal))


# Set up boundary conditions for collocation
c_Q = np.full(N_total,1/2) 
u_Q = np.ones(N_total)          #start potential is 1 and environment potential is unknown
u_Q[N_total-N_startgoal:] = 0   #goal potential
gam_Q = np.zeros(N_total)       #environment normal derivative is 0
gam_Q[N_environment : ] = 1     #unknown normal derivative on start and goal
u_indices = np.arange(N_environment, N_total)    #indices where we know u
gam_indices = np.arange(0,N_environment)         #indices where we know gam


#ipdb.set_trace()
# Set up matrix A (unknowns) 
A = np.zeros((N_total,N_total))
for i in range(N_total):
    for k in range(N_environment):   #when integrating over environment segment
        is_self = True if i == k else False
        A[i,k] = intDgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1], is_self)
        if is_self:
            A[i,k] += c_Q[i]
    for l in range(N_startgoal):    #integration over start segment
        is_self = True if i == N_environment+l else False
        A[i,N_environment+l] = intgreens(p_all[i], t_startgoal[l], t_startgoal[l+1], center_start, radii_startgoal, is_self)
    for h in range(N_startgoal):    #integration over goal segment
        is_self = True if i == N_environment+N_startgoal+h else False
        A[i,N_environment+N_startgoal+h] = intgreens(p_all[i], t_startgoal[h], t_startgoal[h+1], center_goal, radii_startgoal, is_self)
  
            
# Set up matrix b (knowns)
b = np.zeros(N_total)
for i in range(N_total):
    for k in range(N_environment):
        is_self = True if i ==k else False
        b[i] += gam_Q[k] * intgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, is_self)
    for l in range(N_startgoal):    #integration over start segment
        is_self = True if i == N_environment+l else False
        b[i] += u_Q[N_environment+l] * intDgreens(p_all[i], t_startgoal[l], t_startgoal[l+1], center_start, radii_startgoal, start[l], start[l+1], is_self)     
        if is_self:
            b[i] += - u_Q[i]*c_Q[i]
    for h in range(N_startgoal):    #integration over goal segment
        is_self = True if i == N_environment+N_startgoal+h else False                
        b[i] += u_Q[N_environment + N_startgoal + h] * intDgreens(p_all[i], t_startgoal[h], t_startgoal[h+1], center_goal, radii_startgoal, goal[h], goal[h+1], is_self)
        if is_self:
            b[i] += - u_Q[i]*c_Q[i]
            
            
# Solve matrix equation 
coeff = la.solve(A,b) 
gam_Q[u_indices] = coeff[u_indices]
u_Q[gam_indices] = coeff[gam_indices]      

#Save coefficients elsewhere
gam_Q_environment = gam_Q.copy()
u_Q_environment = u_Q.copy()


# Plot collocation boundary & normals to verify
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(environment[:,0], environment[:,1], 'ro')
plt.plot(p_environment[:,0], p_environment[:,1], 'bo')
plt.axis('equal')
normals1 = np.zeros((N_environment,2))
normals2 = np.zeros((N_startgoal,2))
for i in range(N_environment):
    normals1[i] = find_normal(environment[i], environment[i+1])
for i in range(N_startgoal):
    normals2[i] = find_normal(start[i], start[i+1])
ax.quiver(p_environment[:,0],p_environment[:,1], normals1[:,0], normals1[:,1], color ='r') #U and V are the x and y components of the normal vectors
ax.quiver(p_start[:,0],p_start[:,1], normals2[:,0], normals2[:,1], color ='r')
ax.quiver(p_goal[:,0],p_goal[:,1], normals2[:,0], normals2[:,1], color ='r')

# Plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients
fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)
u_bound_est = [] 
for i in range(N_environment):
    temp = 0
    for k in range(N_environment):
        is_self = True if i == k else False
        temp += 2*(gam_Q[k]*intgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, is_self) - u_Q[k]*intDgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1], is_self))
    for k in range(N_startgoal):
        is_self = True if i == N_environment+k else False
        temp += - 2*(gam_Q[N_environment+k]*intgreens(p_all[i], t_startgoal[k], t_startgoal[k+1], center_start, radii_startgoal, is_self) - u_Q[N_environment+k]*intDgreens(p_all[i], t_startgoal[k], t_startgoal[k+1], center_start, radii_startgoal, start[k], start[k+1], is_self))
    for k in range(N_startgoal):
        is_self = True if i == N_environment+N_startgoal+k else False
        temp += -2* (gam_Q[N_environment+N_startgoal+k]*intgreens(p_all[i], t_startgoal[k], t_startgoal[k+1], center_goal, radii_startgoal, is_self) - u_Q[N_environment+N_startgoal+k]*intDgreens(p_all[i], t_startgoal[k], t_startgoal[k+1], center_goal, radii_startgoal, goal[k], goal[k+1], is_self))
    u_bound_est.append(temp)
ax.plot(list(p_environment[:,0]), list(p_environment[:,1]), u_bound_est, color='blue')
plt.plot(environment[:,0], environment[:,1], 'ro')


# =============================================================================
# # Solve for points inside the domain and plot potential inside domain
# n = 5; r = np.array(radii_startgoal)+[0.001,0.001]; c = center_start; ng = ellipse_grid_count(n, r, c); filename = 'myellipse.png' 
# xy_start = ellipse_grid_points(n, r, c, ng).T
# xy_goal = ellipse_grid_points(n, r, center_goal, ng).T
# xy_environment = ellipse_grid_points(2*n, np.array(radii_environment)-[0.001,0.001], center_environment, ellipse_grid_count(2*n, np.array(radii_environment)-[0.001,0.001], center_environment)).T
# xy = outside_ellipse(xy_environment, center_start, radii_startgoal)
# xy = outside_ellipse(xy, center_goal, radii_startgoal)
# #ellipse_grid_display( n, r, c, ng, xy_horizontal, filename )
# phi = np.zeros((len(xy),1))
# for i in range(len(xy)):                           #There will not be self-terms on the interior
#     phi[i] = potential_at_point_in_workspace(xy[i], u_Q, gam_Q, N_environment, N_startgoal, t_environment, center_environment, radii_environment, environment, t_startgoal, center_start, center_goal, radii_startgoal, start, goal)
# plot3D(xy[:,0],xy[:,1], phi[:,0])  
# =============================================================================





#%% Repeat for obstacle domain *********************************************************
# Set Parameters   
N_obstacle = 20
N_total = N_environment + N_obstacle 
center_obstacle = [1.1, 1.1]         #ellipse center
radii_obstacle = [1.1, 1.1]          #major and minor radius of ellipse

# Discretize outer boundary
obstacle, t_obstacle = ellipse_collocation(N_obstacle+1, center_obstacle, radii_obstacle)
t_p_obstacle = np.absolute(t_obstacle[:-1] + t_obstacle[1:])/2  
p_obstacle = ellipse_knots(N_obstacle+1, center_obstacle, radii_obstacle) 
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
        A[i,k] = intgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, is_self)
    for l in range(N_obstacle):
        is_self = True if i == N_environment+l else False
        A[i,N_environment+l] = - intgreens(p_all[i], t_obstacle[l], t_obstacle[l+1], center_obstacle, radii_obstacle, is_self)

# Set up matrix b (knowns)
b = np.zeros(N_total)
for i in range(N_total):
    for k in range(N_environment):
        is_self = True if i == k else False
        b[i] +=  u_Q[k]*intDgreens(p_all[i], t_environment[k], t_environment[k+1], center_environment, radii_environment, environment[k], environment[k+1], is_self)
        if is_self:
            b[i] +=  u_Q[i]*c_Q[i]
    for l in range(N_obstacle):
        is_self = True if i == N_environment+l else False
        b[i] +=  - u_Q[N_environment+l]*intDgreens(p_all[i], t_obstacle[l], t_obstacle[l+1], center_obstacle, radii_obstacle, obstacle[l], obstacle[l+1], is_self)
        if is_self:
            b[i] += u_Q[i]*c_Q[i]

            
# Solve matrix equation for gam on environment and obstacle
gam_Q = la.solve(A,b) 
print(u_Q)
print(gam_Q)

#===========================================================================
    # Generate Plots #
#===========================================================================

# =============================================================================
# 
# # Solve for points inside the domain and plot potential inside domain
# n = 6; r = np.array(radii_obstacle)+[0.001,0.001]; c = center_obstacle; ng = ellipse_grid_count(n, r, c); filename = 'myellipse2.png' 
# xy_inner = ellipse_grid_points(n, r, c, ng).T
# xy_outer = ellipse_grid_points(2*n, np.array(radii_environment)-[0.001,0.001], center_environment, ellipse_grid_count(2*n, np.array(radii_environment)-[0.001,0.001], center_environment)).T
# xy = outside_ellipse(xy_outer, c, r)
# #ellipse_grid_display( 2*n+1, 2*r, c, ellipse_grid_count(2*n+1, 2*r, c) , xy_outer, filename )
# phi = np.zeros((len(xy),1))
# for i in range(len(xy)):    
#     phi[i] = potential_at_point_in_obstacle_domain(xy[i], u_Q, gam_Q, N_environment, N_obstacle, t_environment, center_environment, radii_environment, environment, t_obstacle, center_obstacle, radii_obstacle, obstacle)
# plot3D(xy[:,0],xy[:,1], phi[:,0]) 
# 
# =============================================================================

#%% Generate path


def calculate_field( p, u, gam, N_env, N_sg, t_env, c_env, r_env, env, t_sg, c_s, c_g, r_sg, start, goal, dx = 0.05, dy = 0.05):
    U2 = potential_at_point_in_workspace(p + [dx, 0.0], u, gam, N_env, N_sg, t_env, c_env, r_env, env, t_sg, c_s, c_g, r_sg, start, goal)
    U4 = potential_at_point_in_workspace(p - [dx, 0.0], u, gam, N_env, N_sg, t_env, c_env, r_env, env, t_sg, c_s, c_g, r_sg, start, goal)
    U1 = potential_at_point_in_workspace(p + [0.0, dy], u, gam, N_env, N_sg, t_env, c_env, r_env, env, t_sg, c_s, c_g, r_sg, start, goal)
    U3 = potential_at_point_in_workspace(p - [0.0, dy], u, gam, N_env, N_sg, t_env, c_env, r_env, env, t_sg, c_s, c_g, r_sg, start, goal)                                     
    E = - np.array( [(U2 - U4) / (2*dx), (U1 - U3) / (2*dy)] )
    return E
    
def is_near_goal(p, goal):
    return la.norm(p - goal) <= 0.1
    
#ipdb.set_trace()
step_size = 0.06
obstacle_threshold = 1.0
trajectory = [np.array(center_start)]
trajectory.append(center_start + np.array([step_size, step_size]) )
while not is_near_goal(trajectory[-1], center_goal):
    Uob = potential_at_point_in_obstacle_domain(trajectory[-1], u_Q, gam_Q, N_environment, N_obstacle, t_environment, center_environment, radii_environment, environment, t_obstacle, center_obstacle, radii_obstacle, obstacle)
    Efield = calculate_field(trajectory[-1], u_Q_environment, gam_Q_environment, N_environment, N_startgoal, t_environment, center_environment, radii_environment, environment, t_startgoal, center_start, center_goal, radii_startgoal, start, goal)
    Efield_normalized = Efield/la.norm(Efield)
    if Uob > obstacle_threshold:
        print('follow contour')
        Field_perp_to_E = np.array([-Efield[1], Efield[0]])
        trajectory.append( trajectory[-1] + step_size * Field_perp_to_E ) 
        #while vector in -- keep following  statement
    else:  
        trajectory.append( trajectory[-1] + step_size * Efield_normalized ) 
    

#%%
 # Plot collocation boundary & normals to verify
traj = np.vstack(trajectory)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(traj[:,0], traj[:,1], 'ro')

plt.plot(environment[:,0], environment[:,1], 'ro')
plt.plot(p_environment[:,0], p_environment[:,1], 'bo')

normals1 = np.zeros((N_environment,2))
normals2 = np.zeros((N_startgoal,2))
for i in range(N_environment):
    normals1[i] = find_normal(environment[i], environment[i+1])
for i in range(N_startgoal):
    normals2[i] = find_normal(start[i], start[i+1])
ax.quiver(p_environment[:,0],p_environment[:,1], normals1[:,0], normals1[:,1], color ='r') #U and V are the x and y components of the normal vectors
ax.quiver(p_start[:,0],p_start[:,1], normals2[:,0], normals2[:,1], color ='r')
ax.quiver(p_goal[:,0],p_goal[:,1], normals2[:,0], normals2[:,1], color ='r')
        
        
    
    


    
    
    
    
    
    
    
    
    
    
     
