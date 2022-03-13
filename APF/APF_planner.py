#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:30:24 2022

#Test script to run IEM path planning algorithm.

@author: ubuntu
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import ipdb
import cprofiler
from meshEllipse import *
from IEM_ellipse import *

##################### FUNCTIONS #############################################################
    
def is_near_goal(p, goal, step):
    return la.norm(p - goal) <= step

def generate_random_movement(obstacle, base):
    while True:
        delta_pos = np.random.sample((5,))
        if outside_ellipse(obstacle, base):
            continue
        else:
            obstacle.center += delta_pos
            break
        
def outside_ellipse(ob, base):
    '''pos is list of positions Xx2. returns Xx2 array of positions that are
    outside the ellipse with center and radius c and r'''
    inequality = (ob.points[:,0]-base.center[0])**2/base.radii[0]**2 + (ob.points[:,1]-base.center[1])**2/base.radii[1]**2
    is_outside = (inequality > 1).any()
    return is_outside

def inside_base_ellipse(xx, yy, c, r):
    '''pos is list of positions Xx2. returns Xx2 array of positions that are
    outside the ellipse with center and radius c and r'''
    inequality = (xx-c[0])**2/r[0]**2 + (yy-c[1])**2/r[1]**2
    is_inside = inequality < 1
    return is_inside

def greens_eigenfunction_expansion_for_circle(center, radii):
    ''' cen is the center [,] and radii is the radii [,]. '''
    a = radii[0]
    xx, yy = np.meshgrid(np.arange(-a,a, 1), np.arange(-a,a,1))
    valid_coords = inside_base_ellipse(xx, yy, center, radii)
    #n = 5; r = np.array([a,a]) - [0.01,0.01]; c = [0,0]; ng = ellipse_grid_count(n, r, c)
    #xy = ellipse_grid_points(n, r, c, ng).T
    #r = np.sqrt(xy[:,0]**2 + xy[:,1]**2)
    #theta = np.zeros(len(r))
    #theta[1:] = np.arccos(xy[1:,0]/r[1:]) 
    
    g_free = np.full((xx.shape[0], xx.shape[1], xx.size), 0.0)   #Nx X Ny X Npoints
    g_scatter = np.full((xx.shape[0], xx.shape[1], xx.size), 0.0)
    
    r = np.sqrt(xx**2 + yy**2)
    r[int(xx.shape[0]/2), int(xx.shape[1]/2)] = 1  #set arbitrary angle for radius of length 0
    theta = np.arccos(xx/r)                                            
    theta[ yy < 0 ] *= -1
    #theta[int(xx.shape[0]/2), int(xx.shape[1]/2)] = 3  #for theta at 0,0
    r[int(xx.shape[0]/2), int(xx.shape[1]/2)] = 0  #set back correct radius
    ipdb.set_trace()
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
                THETA = theta[i,j]; R = r[i,j]
                temp = r[i,j]; r[i,j] = 20 #Temporary switch for self-terms
                g_free[:,:,j+i*xx.shape[1]][valid_coords] = - 1/(4*np.pi) * np.log(r[valid_coords]**2 +R**2 - 2*r[valid_coords]*R*np.cos(theta[valid_coords] - THETA)) #solution to del^2 = -(delta function)
                r[i,j] = temp  #switch back
                g_free[i,j,j+i*xx.shape[1]] = np.inf
                g_scatter[:,:,j+i*xx.shape[1]][valid_coords] = - (1/(2*np.pi) * np.log(a) - 1/(4*np.pi)*np.log(a**4 + R**2*r[valid_coords]**2 - 2*a**2*R*r[valid_coords]*np.cos(theta[valid_coords]-THETA)))
    greens_table = g_free + g_scatter
    xx = xx + center[0] ; yy = yy + center[1]
# =============================================================================
#     fig, axs = plt.subplots(1,3)
#     axs[0].bar(names, values)
#     axs[1].scatter(names, values)
#     axs[2].plot(names, values)
#     fig.suptitle('Categorical Plotting')
# =============================================================================
# =============================================================================
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
#     c = ax.scatter(theta, r)
# =============================================================================
    #plt.polar(theta, r, marker = ".")
    #fig, ax = plt.subplots()
    #circle = plt.Circle(center, a, color='thistle')
    #ax.add_patch(circle)
   # plt.scatter( xx[valid_coords], yy[valid_coords], marker='.' )
    plot3D(xx[valid_coords], yy[valid_coords], g_free[:,:,210][valid_coords], title = 'Free G')
    plot3D(xx[valid_coords], yy[valid_coords], g_scatter[:,:,210][valid_coords], title = 'Scattered G')
    plot3D(xx[valid_coords], yy[valid_coords], greens_table[:,:,210][valid_coords], title = 'Total G')
    
    return xx, yy, g_free, g_scatter, greens_table

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


# =============================================================================
# def interpolate_greens(point, values, radii):
#     ''' point is a [,] 
#         values is a 20x20xN_points array of values at the gridpoints
#         Returns bilinear interpolation of the gridpoints
#     '''
#     new_point = point + radii  #align the ellipse in the first quadrant
#     x1, y1 = np.floor(new_point)
#     x2, y2 = np.ceil(new_point)
#     rectangle = [ (x1,y1,values[]), (x1,y2,values[]), (x2,y1,values[]), (x2,y2,values[]) ]
#     G_at_point = bilinear_interpolation(point[0], point[1], rectangle)
#     return G_at_point
# =============================================================================
    


################## PLANNERS ###################################################################

#@cprofiler.profile()
def APF_planner(base_domain, obstacle_domains):    
    ''' base_domain is a FEM_domain object, obstacle_domain is a list of FEM_objects '''
    base_domain.calculate_potential_field()
    flag = 0
    
    step_size = 0.5
    obstacle_threshold = 0.6
    direction_vectors = []
    direction_vectors.append(np.array([np.sqrt(1/2), np.sqrt(1/2)]) ) #take first step out of start 
    trajectory = [base_domain.start]
    trajectory.append(trajectory[-1] + np.array([step_size, step_size]) ) #take first step out of start
    for obstacle in obstacle_domains:
        obstacle.calculate_potential_field()
    
    while not is_near_goal(trajectory[-1], base_domain.goal, step_size):
        U_base = base_domain.Ufield(trajectory[-1])
        E_base = base_domain.Efield(trajectory[-1])
        Ebase_normalized = E_base/la.norm(E_base)
        
        #ipdb.set_trace()
        for ob in obstacle_domains:
            U_obstacle = ob.Ufield(trajectory[-1])
            if U_obstacle > obstacle_threshold:
                print('Initiate wall-following')
                Uin = U_base
                direction_check = True #check direction of wall following first
                while True:
                    E_obstacle = ob.Efield(trajectory[-1])
                    Eobstacle_normalized = E_obstacle/la.norm(E_obstacle)
                    if U_base < Uin:
                        if np.dot(E_obstacle, E_base) >= 0:
                            break
                        
                    E_perp = np.array([-Eobstacle_normalized[1], Eobstacle_normalized[0]])
                    if direction_check:
                        if np.dot(E_perp, E_base) >= 0:
                            direction = +1
                        else:
                            direction = -1
                        direction_check = False 
                    E_perp = direction * E_perp
                    trajectory.append( trajectory[-1] + step_size * E_perp )
                    
              
                    for other_ob in obstacle_domains: #check for intersection with other obstacles
                        if other_ob == ob:
                            continue
                        if other_ob.Ufield(trajectory[-1]) > obstacle_threshold:
                            #ipdb.set_trace()
                            ob = other_ob
                                              
                    U_base = base_domain.Ufield(trajectory[-1])
                    E_base = base_domain.Efield(trajectory[-1])
                    Ebase_normalized = E_base/la.norm(E_base)
                    #print(trajectory[-1])
                    flag +=1
                    if flag > 1000:
                        print('Failure')
                        break

        trajectory.append( trajectory[-1] + step_size * Ebase_normalized ) 
        #ipdb.set_trace()
        #debugging
        flag +=1
        if flag > 100:
            print('Failure')
            break
        if trajectory[-1].any() > 3 or trajectory[-1].any() < -3:
            print('Error: Trajectory out of bounds')
            break
       # print(trajectory[-1])
    return trajectory


def APF_planner2(base_domain, obstacle_domains):    
    ''' This planner samples the boundary Green's function for use in obstacle domain calculations.
    '''
    base_domain.calculate_potential_field()
    #xx, yy, G_free, G_scatter, G_total = greens_eigenfunction_expansion_for_circle(base_domain.center, base_domain.radii)
    flag = 0
    
    step_size = 0.5
    obstacle_threshold = 0.6
    direction_vectors = []
    direction_vectors.append(np.array([np.sqrt(1/2), np.sqrt(1/2)]) ) #take first step out of start 
    trajectory = [base_domain.start]
    trajectory.append(trajectory[-1] + np.array([step_size, step_size]) ) #take first step out of start
    for obstacle in obstacle_domains:
        obstacle.calculate_potential_field()
        
    #test_G_scattered(obstacle_domains[0])
    
    while not is_near_goal(trajectory[-1], base_domain.goal, step_size):
        U_base = base_domain.Ufield(trajectory[-1])
        E_base = base_domain.Efield(trajectory[-1])
        Ebase_normalized = E_base/la.norm(E_base)
        
        #ipdb.set_trace()
        for ob in obstacle_domains:
            U_obstacle = ob.Ufield(trajectory[-1])
            if U_obstacle > obstacle_threshold:
                print('Initiate wall-following')
                Uin = U_base
                direction_check = True #check direction of wall following first
                while True:
                    E_obstacle = ob.Efield(trajectory[-1])
                    Eobstacle_normalized = E_obstacle/la.norm(E_obstacle)
                    if U_base < Uin:
                        if np.dot(E_obstacle, E_base) >= 0:
                            break
                        
                    E_perp = np.array([-Eobstacle_normalized[1], Eobstacle_normalized[0]])
                    if direction_check:
                        if np.dot(E_perp, E_base) >= 0:
                            direction = +1
                        else:
                            direction = -1
                        direction_check = False 
                    E_perp = direction * E_perp
                    trajectory.append( trajectory[-1] + step_size * E_perp )
                    
              
                    for other_ob in obstacle_domains: #check for intersection with other obstacles
                        if other_ob == ob:
                            continue
                        if other_ob.Ufield(trajectory[-1]) > obstacle_threshold:
                            #ipdb.set_trace()
                            ob = other_ob
                                              
                    U_base = base_domain.Ufield(trajectory[-1])
                    E_base = base_domain.Efield(trajectory[-1])
                    Ebase_normalized = E_base/la.norm(E_base)
                    #print(trajectory[-1])
                    flag +=1
                    if flag > 100:
                        print('Failure')
                        break

        trajectory.append( trajectory[-1] + step_size * Ebase_normalized ) 
        #ipdb.set_trace()
        #debugging
        flag +=1
        if flag > 100:
            print('Failure')
            break
        if trajectory[-1].any() > 3 or trajectory[-1].any() < -3:
            print('Error: Trajectory out of bounds')
            break
       # print(trajectory[-1])
    return trajectory


def APF_planner3(base_domain):    
    ''' This planner computes only one potential field.
    '''
    base_domain.calculate_potential_field()
    flag = 0
    
    step_size = 0.2
    direction_vectors = []
    direction_vectors.append(np.array([np.sqrt(1/2), np.sqrt(1/2)]) ) #take first step out of start 
    trajectory = [base_domain.start]
    trajectory.append(trajectory[-1] + np.array([step_size, step_size]) ) #take first step out of start

    while not is_near_goal(trajectory[-1], base_domain.goal, step_size):
        U_base = base_domain.Ufield(trajectory[-1])
        E_base = base_domain.Efield(trajectory[-1], dx = 0.1, dy = 0.1)
        Ebase_normalized = E_base/la.norm(E_base)
        print(U_base)
        print(E_base)
    
        trajectory.append( trajectory[-1] + step_size * Ebase_normalized ) 
        #ipdb.set_trace()
        #debugging
        flag +=1
        if flag > 50:
            print('Failure')
            break
        if trajectory[-1].any() > 3 or trajectory[-1].any() < -3:
            print('Error: Trajectory out of bounds')
            break
       # print(trajectory[-1])
    return trajectory

    
    