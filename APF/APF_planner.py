#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:30:24 2022

#Test script to run IEM path planning algorithm.

@author: ubuntu
"""

import numpy as np
import numpy.linalg as la
import ipdb

##################### FUNCTIONS #############################################################
    
def is_near_goal(p, goal):
    return la.norm(p - goal) <= 0.1

    #Start with algorithm without greens

def APF_planner(base_domain, obstacle_domains):    
    ''' base_domain is a FEM_domain object, obstacle_domain is a list of FEM_objects '''
    base_domain.calculate_potential_field()
    Nob = len(obstacle_domains) 
    
    step_size = 0.06
    obstacle_threshold = 0.8
    direction_vectors = []
    direction_vectors.append(np.array([np.sqrt(1/2), np.sqrt(1/2)]) ) #take first step out of start 
    trajectory = [base_domain.start]
    trajectory.append(trajectory[-1] + np.array([step_size, step_size]) ) #take first step out of start
    for obstacle in obstacle_domains:
        obstacle.calculate_potential_field()
    
    while not is_near_goal(trajectory[-1], base_domain.goal):
        U_base = base_domain.Ufield(trajectory[-1])
        E_base = base_domain.Efield(trajectory[-1])
        Ebase_normalized = E_base/la.norm(E_base)
        
       # ipdb.set_trace()
        for ob in obstacle_domains:
            U_obstacle = ob.Ufield(trajectory[-1])
            if U_obstacle > obstacle_threshold:
                print('Initiate wall-following')
                Uin = U_base
                while True:
                    E_obstacle = ob.Efield(trajectory[-1])
                    if U_base < Uin:
                        if np.dot(E_obstacle, E_base) >= 0:
                            break
                    Eobstacle_normalized = E_obstacle/la.norm(E_obstacle)
                    E_perp = np.array([-Eobstacle_normalized[1], Eobstacle_normalized[0]])
                    trajectory.append( trajectory[-1] + step_size * E_perp )
                    E_base = base_domain.Efield(trajectory[-1])
                    Ebase_normalized = E_base/la.norm(E_base)
                    print(trajectory[-1])

        trajectory.append( trajectory[-1] + step_size * Ebase_normalized ) 
        #ipdb.set_trace()
        #debugging

        if trajectory[-1].any() > 3 or trajectory[-1].any() < -3:
            print('Error: Trajectory out of bounds')
            break
       # print(trajectory[-1])
    return trajectory

    
    