# -*- coding: utf-8 -*-
"""
Author: Esther Grossman
ECE 222C Project 2 : FEM for robot navigation
Note: To change which mesh you're working with, change name of load text files and
uncomment the appropriate boundary conditions'
To change the trajectory, change the value of the potentials for the goal and for 
the walls. Also change start and end point of trajectory if walls have changed.

"""
import numpy as np
import numpy.linalg as la
import math
pi = math.pi
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from scipy.linalg import null_space
#import trimesh

# Separation of variables solution for Box case from Proj 2
def AnalyticalPhiBox(a,b,phi_o,nodes):
    '''Input is lengths a and b of sides of box, potential of the top box, and
    list of node coordinates. Outputs the analytical solution for all coordinates.'''
    ana=[]
    for i in range(len(nodes)):
        summer=0
        for n in range(1,100):
            c_n=4/(pi*n)*(np.sinh(pi*n*nodes[i,1]/a)/np.sinh(pi*n*b/a))
            summer+=c_n*np.sin(pi*n*nodes[i,0]/a)
        ana.append(summer*phi_o)
    return ana
   
# Function to check if position is in a given element    
def xyInElement(coord,p):
    '''Input is a list of np.arrays containing the coordinates of the vertices 
    of the triangle. p is a numpy array of the current position (x,y).
    Returns True is point is inside or on the edge of the triangle.'''
    AB = coord[1]-coord[0] #vertex of point B - vertex of point A
    AC = coord[2]-coord[0]
    a = (np.cross(p,AB)-np.cross(coord[0],AB))/np.cross(AC,AB)
    b = -(np.cross(p,AC)-np.cross(coord[0],AC))/np.cross(AC,AB)
    if a >= 0 and b >= 0 and (a+b) < 1:
        return True
    else:
        return False
    
# Function to find the E_field vector in a given element (constant for each element)
def Efield(nodes,A,e_phi):
    '''Input a numpy array of the coordinates of the three vertices of the element.
    A is the area of the element. e_phi is a numpy array of the potentials at each
    of the three vertices. Returns the electric field vector as a numpy array [x,y].'''
    [v1, v2, v3] = nodes #coords of each node
    #a=[v2[0]*v3[1]-v2[1]*v3[0], v3[0]*v1[1]-v3[1]*v1[0], v1[0]*v2[1]-v1[1]*v2[0]] 
    b=np.array([v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]])
    c=np.array([v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]])
    E=-1/(2*A)*np.array([sum(b*e_phi),sum(c*e_phi)])   #x and y coordinates of E field
    return E
    #potential interpolation
    #N=np.zeros[3]
    #for j in range(3):
    #    N[j]=1/(2*e_area)*(a[j]+b[j]*x+c[j]*y)
    #phi = sum(N*e_phi)
   
# Solve for the local K matrix of a given element
def get_Ke(elements,nodes,e,A):
    '''Input array of element indices, array of node coordinates, element number,
    and element area. Returns 3x3 Ke array for that element.'''
    assert isinstance(e,int)
    [n1, n2, n3] = elements[e] #node numbers of element e
    [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
    b=[v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]]
    c=[v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]]
    Ke=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Ke[i,j] = 1/(4*A)*(b[i]*b[j]+c[i]*c[j])
    return Ke

#=============================================================================
    # MAIN CODE #
#=============================================================================
    
# Import node coordinates and connectivity matrix
elements = np.loadtxt(fname ='elements_box.txt',delimiter = ",",dtype=int)
nodes = np.loadtxt(fname = 'node_coord_box.txt',delimiter = ",")
bounds = np.loadtxt(fname ='boundaries_box.txt',dtype=int) #indices of the boundary nodes
goals = np.loadtxt(fname ='goal_box.txt',dtype=int) #indices of the boundary nodes
#mesh=trimesh.Trimesh(nodes,elements-1,process=False); mesh.show()

# Calculate the area of each element
element_area=[]
for i in elements:
    AB = nodes[i[1]-1]-nodes[i[0]-1] #vertex of point B - vertex of point A
    AC = nodes[i[2]-1]-nodes[i[0]-1] #vertex of point C - vertex of point A
    element_area.append(abs(np.cross(AB, AC))*1/2)
    
# Solve the elemental/local functional equation for every element
num_e = len(elements)
num_phi = len(nodes)
K_local = np.zeros((num_e,3,3))
K_global = np.zeros((num_phi,num_phi))
b_global = np.zeros(num_phi)

for el in range(num_e):
    K_local[el,:,:]=get_Ke(elements,nodes,el,element_area[el])
 
# Place local K into global matrix
for el in range(num_e):
    for i in range(3):
        ni=elements[el,i]-1
        for j in range(3):
            nj=elements[el,j]-1
            K_global[ni,nj] += K_local[el,i,j]
         
# Impose Boundary Conditions
#                              ***for box potential***
bound=[(value-1) for value in bounds if value not in goals]  # indices of boundaries at a 1st potential
goal=np.array([goals-1])    #boundaries at a 2nd potential
bounds = bounds -1 
# =============================================================================
# #                        ***for small 9-point test case***
# bound=np.array([1,5,6,7,8])   #set which node index are on the boundary
# goal=np.array([0,2,4])                #set which node is goal
# bounds=np.concatenate((bound,goal))
# =============================================================================
# =============================================================================
# #                            ***for matlab circle***
# bound=[(value-1) for value in bounds if value not in goals] #the -1 is becauce indices came from matlab which doesnt start at 0
# goal=np.array([goals-1])  #goal boundaries
# bounds=bounds-1           #all boundaries
# =============================================================================
for n in bound:           #for loop for setting wall boundary conditions
    K_global[n,n]=1
    b_global[n]=10        #value of potential at boundaries/for box is 0
    for i in range(0,num_phi):
        if i != n:
            K_global[n,i]=0 
        if i not in bounds:   #check all boundaries
            b_global[i]-=K_global[i,n]*b_global[n]  
            K_global[i,n]=0
for m in goal:            
    K_global[m,m]=1   #setting goal boundary condition
    b_global[m]=1 #value of potential at boundaries/for box is phi_0
    for i in range(0,num_phi):
        if i != m:
            K_global[m,i]=0   
            if i not in bounds:
                b_global[i]-=K_global[i,m]*b_global[m]
                K_global[i,m]=0
#print(K_global)
            
# =============================================================================
# #Quick Boundary Conditions - not working
# K_global[0,0]=10^70
# b_global[0]=1*10^70
# K_global[5,5]=10^70
# b_global[5]=2*10^70
# =============================================================================

# Solve matrix
phi = la.solve(K_global,b_global)
 #Compare to analytical (for box case)
Ana_phi=AnalyticalPhiBox(2,2,10,nodes)        

# =============================================================================
# # Solve for trajectory of robot
# start_pos=[10.5,10]
# pos=np.array(start_pos) #start position
# path=[start_pos]
# end_pos=np.array([33.71,20.74]) #end position is upper wall, so y=2
# tol=1
# step=0.5
# while la.norm(pos-end_pos) > tol:  #use when a single coordinate is goal
# #while abs(pos[1]-end_pos) > tol:   #use when an entire wall is the goal
#     count = 0
#     for i in elements:
#         if xyInElement([nodes[i[0]-1],nodes[i[1]-1],nodes[i[2]-1]],pos):
#             E = Efield(np.array([nodes[i[0]-1],nodes[i[1]-1],nodes[i[2]-1]]), element_area[count], np.array([phi[i[0]-1],phi[i[1]-1],phi[i[2]-1]]))
#             break
#     pos += E/la.norm(E)*step
#     path.append(pos.tolist())
#     count += 1
# 
# # Plot trajectory
# fig=plt.figure()
# ax = fig.add_subplot(111)
# for x,y in path:
#     ax.plot(x,y, marker="o", color="blue", markeredgecolor="black") 
#     #print(x,y)
# for i in bounds:
#     ax.plot(nodes[i,0],nodes[i,1], marker='X')
# plt.axis([5, 35, 10, 25]) 
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)
#     ax.spines[axis].set_color("orange")
# plt.show()
#     
# =============================================================================

#solve for electric field vector in each element
#start robot at wall position
#move robot in direction of electric field for 1 timestep
#locate element at new position
#move in the direction of the field at the point for 1 timestep
#repeat until goal is reached


                   

