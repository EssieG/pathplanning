#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:52:54 2022

@author: ubuntu
"""

import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Ellipse
import ipdb
from FEM_domain import FEM_domain
from IEM_ellipse import IEM_ellipse
from APF_planner import *

''' Use this to measure speed of all functions: python -m cProfile -s cumtime main.py '''

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  

def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def runtest(base_domain, obstacle_domains, start, goal, verbose = True):
  '''
  This function:
   * load the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''
 
  boundary = [[0, 0, 0, 2, 2, 0, 200, 200, 200]]
  blocks = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]  
# =============================================================================
#   # Display the environment
#   if verbose:
#     fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)  
# =============================================================================
  # Display the environment
  
  # Call the motion planner
  t0 = tic()
  #path = APF_planner2(base_domain, obstacle_domains) #TODO: Path Planner
  path = APF_planner2(base_domain, obstacle_domains)
  pathArray = np.array(path)
  toc(t0,"Planning")
  
# =============================================================================
#   # Plot the path
#   if verbose:
#     ax.plot(path[:,0],path[:,1],path[:,2],'r-', marker = 'o')
# =============================================================================

  if verbose:
      fig, ax = plt.subplots()
      plt.title('Trajectory')
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)   
      ax.spines['bottom'].set_visible(False)
      ax.spines['left'].set_visible(False)
      boundary_points = base_domain.get_boundary_points()
      ax.plot(boundary_points[:,0], boundary_points[:,1])
      for ob in obstacle_domains:   #for planner1 and 2
          boundary_points = ob.get_boundary_points()
          ax.plot(boundary_points[:,0], boundary_points[:,1], color = 'r')
# =============================================================================
#       for ob in obstacle_domains:
#           ellipse = Ellipse(ob[0], 2*ob[1][0], 2*ob[1][1], color='thistle')
#           ax.add_patch(ellipse)
#       ellipse = Ellipse(base_domain.start, 2*base_domain.startgoal_radii[0], 2*base_domain.startgoal_radii[1], color='pink') #for planner 3
#       ax.add_patch(ellipse)
#       ellipse = Ellipse(base_domain.goal, 2*base_domain.startgoal_radii[0], 2*base_domain.startgoal_radii[1], color='pink')
#       ax.add_patch(ellipse)
# =============================================================================
      ax.plot(pathArray[:,0], pathArray[:,1], linestyle='--', marker='o', color='b')
      ax.plot(base_domain.start[0], base_domain.start[1], marker='X', color='g', markersize = 12)
      ax.plot(base_domain.goal[0], base_domain.goal[1], marker='*', color='y', markersize = 12)
 
      plt.show()

  collision = False #collision_check(path, blocks[:])      #False if no collision
  goal_reached = sum((path[-1]-goal)**2) <= 0.1
  success = (not collision) and goal_reached
  pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
  return success, pathlength


def collision_check(path, blocks):
    ''' This is a subroutine used by runtest to check for collisons between the path and 
    an axis-aligned set of bounding boxes (AABB). Segments not allowed to graze box. '''
    p1 = path[:-1]                                       #endpoints of segments
    p2 = path[1:]
    #ipdb.set_trace()
    collisionStillPossible = np.full((len(p1),len(blocks)), True)
    for dim in range(3):                                 #check all three dimensions
        D = np.hstack((p1[:, dim,None], p2[:,dim,None])) #D is dim coordinate of the two endpoints
        minD = np.repeat(np.amin(D, 1)[:,None],len(blocks), axis=1); maxD = np.repeat(np.amax(D, 1)[:,None],len(blocks),axis=1) #column vectors of the max and min for each seg, NxB
        maxswap = maxD > blocks[:, 3 + dim]              #blocks should be dimension (b,)
        maxD[maxswap] = np.repeat(blocks[:, 3+dim][:,None], len(p1), axis=1).T[maxswap]   
        minswap = minD < blocks[:, dim]
        minD[minswap] = np.repeat(blocks[:, dim][:,None], len(p1), axis=1).T[minswap]
        collisionStillPossible = np.logical_and(collisionStillPossible, minD <= maxD)  #N x B

    return collisionStillPossible.any()                  #collision must have occured

# =============================================================================
# # Plot trajectory
#  fig=plt.figure()
#  ax = fig.add_subplot(111)
#  for x,y in path:
#      ax.plot(x,y, marker="o", color="blue", markeredgecolor="black") 
#      #print(x,y)
#  for i in bounds:
#      ax.plot(nodes[i,0],nodes[i,1], marker='X')
#  plt.axis([5, 35, 10, 25]) 
#  for axis in ['top','bottom','left','right']:
#      ax.spines[axis].set_linewidth(2)
#      ax.spines[axis].set_color("orange")
#  plt.show()
# =============================================================================

######################################## ENVIRONMENTS #########################
  
def test_small_square():
     print('Running test on 9-point square...\n')
     start = np.array([0.1, 0.1])
     goal = np.array([1.55, 1.55])
     elements = np.loadtxt(fname ='elements_test.txt',delimiter = ",",dtype=int)
     nodes = np.loadtxt(fname = 'node_coord_test.txt',delimiter = ",") + np.ones(2)  #lower left corner is origin.
     base_domain = FEM_domain(elements, nodes, 'mixed', start, goal)
     obstacle_domains = []
     obstacle_domains.append(FEM_domain(elements, nodes, 'dirichlet'))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     #print('Success: %r'%success)
     #print('Path length: %d'%pathlength)
     print('\n')
     return 

def test_square():
     print('Running test on square...\n')
     start = np.array([0.1, 0.1])
     goal = np.array([1.55, 1.55])
     elements = np.loadtxt(fname ='elements_box.txt', delimiter = ",",dtype=int)
     nodes = np.loadtxt(fname = 'node_coord_box.txt', delimiter = ",") #lower left corner is origin.
     base_domain = FEM_domain(elements, nodes, 'mixed', start, goal)
     obstacle_domains = []
     obstacle_domains.append(FEM_domain(elements, nodes, 'dirichlet'))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return
  
def test_circle():
     print('Running test on circle...\n')
     start = np.array([-0.9, -0.2])
     goal = np.array([0.4, 0.44])
     boundaries = np.loadtxt(fname ='boundaries.txt', delimiter = ",",dtype=int)
     elements = np.loadtxt(fname ='elements.txt', delimiter = ",",dtype=int)
     nodes = np.loadtxt(fname = 'node_coord.txt', delimiter = ",") #lower left corner is origin.
     base_domain = FEM_domain(elements, nodes, 'mixed', start, goal)
     obstacle_domains = []
     obstacle_domains.append(FEM_domain(elements, nodes, 'dirichlet'))
     #Plot nodes
     fig, ax = plt.subplots()
     plt.title('Trajectory')
     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.scatter(base_domain.nodes[:,0], base_domain.nodes[:,1])
     plt.show()
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return   

def test_IEM_oneCircle():
     print('Running test on double circle...\n')
     start = np.array([0.0, 0.0])
     goal = np.array([2.0, 2.0])
     center = np.array([1, 1])
     radii = np.array([2, 2])
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_ellipse(np.array([1.1,1.1]), np.array([1.1,1.1]), 'dirichlet', N = 20, other = base_domain))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return   
 
def test_IEM_threeCircle():
     print('Running test on triple circle...\n')
     start = np.array([-5,-5])
     goal = np.array([7.5,5])
     center = np.array([0, 0])
     radii = np.array([10, 10])
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_ellipse(np.array([1.1,1.1]), np.array([3,3]), 'dirichlet', N = 15, other = base_domain))
     obstacle_domains.append(IEM_ellipse(np.array([2.5,-5]), np.array([2,2]), 'dirichlet', N = 15, other = base_domain))
     obstacle_domains.append(IEM_ellipse(np.array([-4,5]), np.array([4, 2]), 'dirichlet', N = 15, other = base_domain))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return   
 
def test_IEM_threeCircle_static():
     print('Running test on triple circle...\n')
     start = np.array([-5,-5])
     goal = np.array([7.5,5])
     center = np.array([0, 0])
     radii = np.array([10, 10])
     obstacles = []
     obstacles.append((np.array([1.1,1.1]), np.array([3,3])))
     obstacles.append((np.array([2.5,-5]), np.array([2,2])))
     obstacles.append((np.array([-4,5]), np.array([4, 2])))
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal, ob_list = obstacles)
     success, pathlength = runtest(base_domain, obstacles, base_domain.start, base_domain.goal)
     return  
 
def test_IEM_threeCircle_Greens():
     print('Running test on triple circle...\n')
     start = np.array([-5,-5])
     goal = np.array([7.5,5])
     center = np.array([0, 0])
     radii = np.array([10, 10])
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_ellipse(np.array([1.1,1.1]), np.array([3,3]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([2.5,-5]), np.array([2,2]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([-4,5]), np.array([4, 2]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return   
    
  
##################################### MAIN ####################################

if __name__=="__main__":
  #test_square()
  #test_circle()
  #test_small_square()
  #test_IEM_oneCircle()
  #test_IEM_threeCircle()
  #test_IEM_threeCircle_static()
  test_IEM_threeCircle_Greens()
  
  #plt.show(block=False)

  #Plot trajectory
  









