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
#from IEM_ellipse import IEM_ellipse
#from IEM_anyshape2_vector import *
from IEM_anyshape2_vector_allpulse import *
from APF_planner import *
import imageio
import os

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

def plot_animation(trajectory, base, obs ):
    # frames between transitions
    n_frames = 1
    print('Creating charts\n')
    filenames = []

    for index in np.arange(0, len(obs[0].movements)-1):
        # get current and next y coordinates
        #y = np.array(ob_trajectories[index])
        #y1 = np.array(ob_trajectories[index+1])
        fig, ax = plt.subplots()
        boundary_points = base.co_points
        ax.plot(boundary_points[:,0], boundary_points[:,1])
        ax.plot(base.start[0], base.start[1], marker='X', color='g', markersize = 12)
        ax.plot(base.goal[0], base.goal[1], marker='*', color='y', markersize = 12)
        # calculate the distance to the next position
        #y_path = y1 - y 
        for i in np.arange(0, n_frames + 1):
            # divide the distance by the number of frames exit
            # and multiply it by the current frame number
            #y_temp = (y + (y_path / n_frames) * i)        # plot
            
            for j in range(len(obs)):   #for planner1 and 2
                boundary_points, _ = obs[j].ellipse_collocation(obs[j].N_points+1, obs[j].movements[index], obs[j].radii)
                ax.plot(boundary_points[:,0], boundary_points[:,1], color = 'r')
            ax.plot(trajectory[0:index,0], trajectory[0:index,1], linestyle='--', marker='o', color='b')
            
            #plt.ylim(0,80)        # build file name and append to list of file names
            filename = f'gifs/frame_{index}_{i}.png'
            filenames.append(filename)        # last frame of each viz stays longer
            if (i == n_frames):
                for i in range(5):
                    filenames.append(filename)        # save img
            fig.savefig(filename)
            plt.close(fig)       
    print('Charts saved\n')# Build GIF
    print('Creating gif\n')
    with imageio.get_writer('gifs/myplot.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')
    print('Removing Images\n')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('DONE')
        


def runtest(base_domain, obstacle_domains, start, goal, verbose = True, motion = False):
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
  precompute_boundary(base_domain, obstacle_domains)
  #ipdb.set_trace()
  t0 = tic()
  path = APF_planner4(base_domain, obstacle_domains)
  toc(t0,"Planning")
  pathArray = np.array(path)
  
# =============================================================================
#   # Plot the path
#   if verbose:
#     ax.plot(path[:,0],path[:,1],path[:,2],'r-', marker = 'o')
# =============================================================================

  if verbose:
      #ipdb.set_trace()
      if motion == True:
          plot_animation(pathArray, base_domain, obstacle_domains)
      
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
      if isinstance( obstacle_domains, list ):
          for ob in obstacle_domains:   #for planner1 and 2
              boundary_points = ob.get_boundary_points()
              ax.plot(boundary_points[:,0], boundary_points[:,1], color = 'r')
      else:
          for pts in obstacle_domains.ob_co_points: #for planner 
              ax.plot(pts[:,0], pts[:,1], color = 'r')
# =============================================================================
#       for ob in obstacle_domains: #for planner 3
#           ellipse = Ellipse(ob[0], 2*ob[1][0], 2*ob[1][1], color='thistle')
#           ax.add_patch(ellipse)
#       ellipse = Ellipse(base_domain.start, 2*base_domain.startgoal_radii[0], 2*base_domain.startgoal_radii[1], color='pink') 
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

def test_IEM_threeCircle_Greens_shifted():
     print('Running test on triple circle...\n')
     start = np.array([5,5])
     goal = np.array([17.5,15])
     center = np.array([10, 10])
     radii = np.array([10, 10])
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_ellipse(np.array([10.1,10.1]), np.array([3,3]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([12.5,5]), np.array([2,2]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([6.5,15]), np.array([3, 2]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_IEM_threeCircle_Greens_shifted_shape():
     print('Running test on triple circle...\n')
     start = np.array([5,5])
     goal = np.array([16.5,15])
     center = np.array([10, 10])
     radii = np.array([10, 10])
     pts = get_ellipse(30+1, center, radii)
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([10.1,10.1]), np.array([2.5,2.5])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([12.5,4]), np.array([3,1.5])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([6.5,13]), np.array([2.5, 2])), 'dirichlet', other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_IEM_manyCircle_Greens():
     print('Running test on triple circle...\n')
     start = np.array([-5,-5])
     goal = np.array([7.5,5])
     center = np.array([0, 0])
     radii = np.array([10, 10])
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_ellipse(np.array([1.1,1.1]), np.array([.8,.8]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([-7.,-2.5]), np.array([.4,.8]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([0.,5.]), np.array([.7, .5]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([5.,-5.]), np.array([.5, .8]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([6.,0.]), np.array([.9, .9]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([-3.,-5.]), np.array([.5, 2.5]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([-4.3,6.]), np.array([.6, 1]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([2.,-6.]), np.array([.6, 1.5]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([4.,5.]), np.array([.7, .9]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_IEM_mazeCircle_Greens():
     print('Running test on triple circle...\n')
     start = np.array([-5,-5])
     goal = np.array([7.5,5])
     center = np.array([0, 0])
     radii = np.array([10, 10])
     base_domain = IEM_ellipse(center, radii, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_ellipse(np.array([5.0,1.0], dtyp), np.array([.5, 6]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([2.0,-1.0]), np.array([.5, 8]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([0.0,1.0]), np.array([.5, 8]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_ellipse(np.array([-3.0,0.0]), np.array([.5, 6]), 'dirichlet', N = 15, other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 

def test_room_one():
     print('Running test on room one ...')
     #ipdb.set_trace()
     start = np.array([5.0,5.0])
     goal = np.array([15.5,15])
     #goal = np.array([17.5,15])
     pts = 20* np.load('room1_finer.npy')
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([10.1,10.1]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([12.5,5]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([6.5,13.5]), np.array([1, 1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([15,6]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([10,13]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([2,6]), np.array([1, 1])), 'dirichlet', other = base_domain, use_Greens = True))
     
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([12.5,13]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([5,7.5]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([8,5]), np.array([1, 1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([7.5,16]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([15.5,15]), np.array([1,1])), 'dirichlet', other = base_domain, use_Greens = True))
     obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([14,7.5]), np.array([1, 1])), 'dirichlet', other = base_domain, use_Greens = True))
     
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_room_two():
     print('Running test on Zaids star ...')
     #ipdb.set_trace()
     start = np.array([6,12])
     goal = np.array([13,4])
     pts = 20* np.load('room2.npy')
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     #obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([10.1,10.1]), np.array([3,3])), 'dirichlet', other = base_domain, use_Greens = True))
     #obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([12.5,5]), np.array([2,2])), 'dirichlet', other = base_domain, use_Greens = True))
     #obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([6.5,15]), np.array([3, 2])), 'dirichlet', other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return   

def test_room_three():
     print('Running test on Zaids star ...')
     #ipdb.set_trace()
     start = np.array([8,8])
     goal = np.array([12,12])
     pts = 20* np.load('room3.npy')
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     #obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([10.1,10.1]), np.array([3,3])), 'dirichlet', other = base_domain, use_Greens = True))
     #obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([12.5,5]), np.array([2,2])), 'dirichlet', other = base_domain, use_Greens = True))
     #obstacle_domains.append(IEM_anyshape(get_ellipse(15, np.array([6.5,15]), np.array([3, 2])), 'dirichlet', other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_room_four():
     '''this domain has divide by zero problems likely from sharp edges'''
     print('Running test on Zaids star ...')
     #ipdb.set_trace()
     start = np.array([3,6])
     #goal = np.array([11,6])
     goal = np.array([15,15.5])
     pts = 20* np.load('room4.npy')
     ob_pts = 20* np.load('obstacle1.npy')
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts, ob_pts[0])), 'dirichlet', other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_room_five():
     '''Nope. this domain has divide by zero problems likely from sharp edges'''
     print('Running test on Zaids star ...')
     #ipdb.set_trace()
     start = np.array([5,3])
     #goal = np.array([11,6])
     goal = np.array([14,12])
     pts = 20* np.load('room5.npy')
     ob_pts = 20* np.load('obstacle1.npy')
     ob_pts[:,1] -= 5
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     #ipdb.set_trace()
     obstacle_domains = []
     obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts, ob_pts[0])), 'dirichlet', other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_room_six():
     '''this domain is great'''
     print('Running test on Zaids star ...')
     #ipdb.set_trace()
     start = np.array([5,3])
     #goal = np.array([11,6])
     goal = np.array([14,12])
     pts = 20* np.load('room6.npy')
     ob_pts = 20* np.load('obstacle1.npy'); ob_2_pts =  20* np.load('obstacle16_10.npy')
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts, ob_pts[0])), 'dirichlet', other = base_domain, use_Greens = True))
     ob_pts[:,1] = ob_pts[:,1]*0.8 - 2
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts, ob_pts[0])), 'dirichlet', other = base_domain, use_Greens = True))
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_2_pts, ob_2_pts[0])), 'dirichlet', other = base_domain, use_Greens = True))
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_one_obstacle_domain():
     '''this domain is great'''
     print('Running test on Zaids star ...')
     #ipdb.set_trace()
     start = np.array([5,3])
     #goal = np.array([11,6])
     goal = np.array([14,12])
     pts = 20* np.load('room6.npy')
     ob_pts = 20* np.load('obstacle1.npy'); ob_2_pts =  20* np.load('obstacle16_10.npy')
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacles = []
     obstacles.append(np.vstack((ob_pts, ob_pts[0])))
     ob_pts[:,1] = ob_pts[:,1]*0.8 - 2
     #obstacles.append(np.vstack((ob_pts, ob_pts[0])))
     #obstacles.append(np.vstack((ob_2_pts, ob_2_pts[0])))
     obstacle_domain = IEM_anyshape( ob_list = obstacles, boundary_type = 'dirichlet', other = base_domain, use_Greens = True)
     success, pathlength = runtest(base_domain, obstacle_domain,  base_domain.start, base_domain.goal)
     return 
 
def test_square():
     '''this domain has divide by zero problems likely from sharp edges'''
     print('Running test on Maze ...')
     #ipdb.set_trace()
     start = np.array([4,4])
     #goal = np.array([11,6])
     goal = np.array([16,15.5])
     pts = 20* np.load('square.npy')
     ob_pts1 = 20* np.load('obstacle_bar8.npy'); ob_pts2 = 20* np.load('obstacle_bar4.npy'); ob_pts3 = 20* np.load('obstacle_bar6.npy')
     ob_pts1[:,1]/=2
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts2, ob_pts2[0])), 'dirichlet', other = base_domain, use_Greens = True))#good too close to edge
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts3, ob_pts3[0])), 'dirichlet', other = base_domain, use_Greens = True)) #good 
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts1, ob_pts1[0])), 'dirichlet', other = base_domain, use_Greens = True)) #good
     success, pathlength = runtest(base_domain, obstacle_domains, base_domain.start, base_domain.goal)
     return 
 
def test_maze():
     '''this domain has divide by zero problems likely from sharp edges'''
     print('Running test on Maze ...')
     #ipdb.set_trace()
     start = np.array([2.5,5])
     #goal = np.array([11,6])
     goal = np.array([16,15.5])
     pts = 20* np.load('maze.npy')
     ob_pts1 = 20* np.load('obstacle_bar8.npy'); ob_pts2 = 20* np.load('obstacle_bar4.npy'); ob_pts3 = 20* np.load('obstacle_bar6.npy')
     ob_pts1[:,1]/=2
     pts = np.vstack((pts, pts[0]))  #boundary points with first point repeated
     base_domain = IEM_anyshape(pts, 'mixed', start = start, goal = goal)
     obstacle_domains = []
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts2, ob_pts2[0])), 'dirichlet', other = base_domain, use_Greens = True))#good too close to edge
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts3, ob_pts3[0])), 'dirichlet', other = base_domain, use_Greens = True)) #good 
     #obstacle_domains.append(IEM_anyshape(np.vstack((ob_pts1, ob_pts1[0])), 'dirichlet', other = base_domain, use_Greens = True)) #good
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
  #test_IEM_threeCircle_Greens()
  #test_IEM_threeCircle_Greens_shifted()
  #test_IEM_threeCircle_Greens_shifted_shape()
  #test_IEM_manyCircle_Greens()
  #test_IEM_mazeCircle_Greens()
  #plt.show(block=False)

  #test_room_one()
  #test_room_two()
  #test_room_three()
  test_room_six()
  #test_maze()
  #test_square()
  
  #test_room_five()

  #Plot trajectory
  
  #test_one_obstacle_domain()









