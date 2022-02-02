#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:30:24 2022

#Test script to run IEM path planning algorithm.

@author: ubuntu
"""

#might need to manage namespacesn could merge the files since they use many of same functions
import * from IEM_ellipse_environment.py
import * from IEM_ellipse_obstacles.py
import np.linalg as la

#Call planners
start = (1,1)
goal = (2,2)

#calculate gradient around current position
path_vector = finite_difference( pos, dx, dy)
path_vector = la.norm(path_vector)
new_pos = 

