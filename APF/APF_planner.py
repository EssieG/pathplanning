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
#from IEM_ellipse import *
from IEM_anyshape2_vector import *

##################### FUNCTIONS #############################################################
    
def is_near_goal(p, goal, step):
    return la.norm(p - goal) <= step

def generate_random_movement(obstacles, max_step, center, radii):
    for ob in obstacles:
        delta_pos = max_step*( 2*np.random.sample(2) - 1 )
        while not inside_base_ellipse(ob.center[0]+delta_pos[0], ob.center[1]+delta_pos[1], center, radii):
            delta_pos = max_step*( 2*np.random.sample(2) - 1 )
        #ipdb.set_trace()
        ob.center += delta_pos
        #Recalculate ob points
        ob.points = ob.ellipse_knots(ob.N_points+1, ob.center, ob.radii)
        ob.movements.append(np.copy(ob.center))
        
        
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



################## PLANNERS ###################################################################


def APF_planner(base_domain, obstacle_domains):    
    ''' Version that does not use greens for each obstacle domain '''
    base_domain.calculate_potential_field()
    flag = 0
    
    step_size = 0.5
    obstacle_threshold = 0.4
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
    The obstacles MOVE.
    '''
    flag = 0   
    base_domain.calculate_potential_field()
    
    step_size = 0.5
    obstacle_threshold = 0.6
    direction_vectors = []
    direction_vectors.append(np.array([np.sqrt(1/2), np.sqrt(1/2)]) ) #take first step out of start 
    trajectory = [base_domain.start]
    trajectory.append(trajectory[-1] + np.array([step_size, step_size]) ) #take first step out of start
    for obstacle in obstacle_domains:
        obstacle.movements.append(np.copy(obstacle.center))
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
                    generate_random_movement(obstacle_domains, step_size, base_domain.center, base_domain.radii)
                    for obstacle in obstacle_domains:
                        obstacle.calculate_potential_field()
        
              
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
        generate_random_movement(obstacle_domains, step_size, base_domain.center, base_domain.radii)
        for obstacle in obstacle_domains:
            obstacle.calculate_potential_field()
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


def precompute_boundary(base_d, obstacle_ds):
    '''Anything that can be precomputed.
    Base domain potential field
    Greens function due to the boundary
    '''
    base_d.calculate_potential_field()
    g_densities, dg_densities = base_d.calculate_greens_densities()
    for obstacle in obstacle_ds:
        obstacle.dg_on_bounds = dg_densities
        obstacle.g_on_bounds = g_densities
    return


#@cprofiler.profile()
def APF_planner4(base_domain, obstacle_domains):    
    ''' This planner samples the boundary Green's function for use in obstacle domain calculations.
    It is a stationary environment algorithm.
    '''
    flag = 0   
    step_size = 0.5
    obstacle_threshold = 0.6
    direction_vectors = []
    direction_vectors.append(np.array([np.sqrt(1/2), np.sqrt(1/2)]) ) #take first step out of start 
    trajectory = [base_domain.start]
    trajectory.append(trajectory[-1] + np.array([step_size, step_size]) ) #take first step out of start
    for obstacle in obstacle_domains:
        obstacle.calculate_potential_field()
    
    print('Finding path ....')
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
                   # print(trajectory[-1])
              
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
       # print(trajectory[-1])
        #ipdb.set_trace()
        #debugging
        flag +=1
        if flag > 100:
            print('Failure')
            break
        if trajectory[-1].any() > 20 or trajectory[-1].any() < 0:
            print('Error: Trajectory out of bounds')
            break
       # print(trajectory[-1])
    return trajectory  
    