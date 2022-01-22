# -*- coding: utf-8 -*-
"""
Author: Esther Grossman
Dste: 7/22/21
ECE 222C Project 2 : Find Greens Function for triangular elements on a domain
using Finite Element Methods. Phi and Greens are used interchangably in this code
as variable names to describe the greens function at a point

"""
import numpy as np
import numpy.linalg as la
import math
pi = math.pi
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from scipy.linalg import null_space
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import trimesh
import ipdb


def plot3D(x,y,z, title):
    '''Plots anything in 3D. Intended for showing potential surface over 2D plane.'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi$')
    plt.show()

def AnalyticalGreens(a, b, nodes, elements, excitation_index):
    '''get the series solution of greens everywhere due to one excitation''' 
    #ipdb.set_trace()
    G=[]
    [n1, n2, n3] = elements[excitation_index[0]] #node numbers of element e
    [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
    x_prime = (v1[0]+v2[0]+v3[0])/3
    y_prime = (v1[1]+v2[1]+v3[1])/3  #coordinates of excitation (lies in the center of an element)
    for i in range(len(nodes)):
        summer=0
        y = nodes[i,1]
        x = nodes[i,0]
        for m in range(1,200):
            W = np.cosh(m*pi/a*y_prime)*np.sinh(m*pi*(y_prime-b)/a) - np.sinh(m*pi/a*y_prime)*np.cosh(m*pi*(y_prime-b)/a)
            if y > y_prime:
                g_n = 2/(m*pi)*np.cosh(m*pi/a*(y-b))*np.cosh(m*pi/a*y_prime)/W*np.cos(m*pi/a*x_prime)
            else:
                g_n = 2/(m*pi)*np.cosh(m*pi/a*y)*np.cosh(m*pi/a*(y_prime-b))/W*np.cos(m*pi/a*x_prime)
                summer += g_n*np.cos(pi*m*nodes[i,0]/a)    
        G.append(summer)
    return G

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

def interpolate_greens(greens, elements, nodes, e, A ):
    ''' once greens has been computed eerywhere, can interpolate for greens at a specific point'''
    [n1, n2, n3] = elements[e] #node numbers of element e
    [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
    center_triangle = np.array([1/3*(v1[0] +v2[0]+v3[0]),1/3*(v1[1] +v2[1]+v3[1])])
    b=np.array([v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]])
    c=np.array([v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]])
    a = np.array([v2[0]*v3[1]-v2[1]*v3[0], v3[0]*v1[1]-v3[1]*v1[0], v1[0]*v2[1]-v1[1]*v2[0]])
    Ne = 1/(2*A)*(a + b*center_triangle[0] + c*center_triangle[1])
    phi = Ne[0]*greens[n1-1] + Ne[1]*greens[n2-1] + Ne[2]*greens[n3-1]
    return phi
    
def get_Be(e, A, excitation_element):
    '''Input array of element indices, array of node coordinates, element number,
    and element area. Returns 3x3 Ke array for that element.'''
    assert isinstance(e, int)
    if e == excitation_element:
        f = 1 #excitation over element
        b = np.full(3, A/3 * f) 
    else:
        b = np.zeros(3)
    return b

def run_environment(elements, nodes, bound, excitation_index, phi_ana):  
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
    b_local = np.zeros((num_e, 3))
    K_global = np.zeros((num_phi,num_phi))
    b_global = np.zeros(num_phi)
    
    for el in range(num_e):
        K_local[el,:,:] = get_Ke(elements,nodes,el,element_area[el])
        b_local[el,:] = get_Be(el, element_area[el], excitation_index)
    
     
    # Place local K into global matrix
    for el in range(num_e):
        for i in range(3):
            ni=elements[el,i]-1
            b_global[ni] += b_local[el,i]
            for j in range(3):
                nj=elements[el,j]-1
                K_global[ni,nj] += K_local[el,i,j]
     
    #ipdb.set_trace()
    #Boundary Conditions
    #Subtract average RHS from RHS so sum is 0 (DISCRETE COMPATIBILITY)
    b_global = b_global - np.average(b_global)
    #Set ones known Greens node (dirichlet) (SPECIFYING CONSTANT)                      
    for n in bound:          
        K_global[n,n] = 1
        b_global[n] = 1        #specify node value
        for i in range(0,num_phi): #Preserve symmetry
            if i != n:
                K_global[n,i] = 0 
            if i not in bound:   
                b_global[i] -= K_global[i,n]*b_global[n]  
                K_global[i,n] = 0
    
    # Solve matrix
    phi = la.solve(K_global,b_global)
    
    # Interpolate to get greens at the center of each node
    greens_at_centers = []
    for el in range(num_e):
        temp = interpolate_greens(phi, elements, nodes, el, element_area[el])
        greens_at_centers.append(temp)
    print('greens at centers FEM:', greens_at_centers)
    
    # Interpolate to get greens at the center of each node for analytical solution
    greens_at_centers = []
    for el in range(num_e):
        temp = interpolate_greens(phi_ana, elements, nodes, el, element_area[el])
        greens_at_centers.append(temp)
    print('greens at centers Series:', greens_at_centers)
    
    return phi

#=============================================================================
    # Environments #
#=============================================================================

# Define environments
def test_small_square():
     print('Running test on 9-point square...\n')
     nodes = np.array([[0., 2.],
                     [0., 1.],
                     [1., 2.],
                     [1., 1.],
                     [2., 2.],
                     [2., 1.],
                     [0., 0.],
                     [1., 0.],
                     [2., 0.]])
     elements = np.array([[2, 4, 1],
                         [4, 3, 1],
                         [4, 6, 3],
                         [6, 5, 3],
                         [7, 8, 2],
                         [8, 4, 2],
                         [8, 9, 4],
                         [9, 6, 4]])
     bound = np.array([3])                 #set index of a "dirichlet node" which will fix greens at a node
     excitation_index = np.array([3])      #set index of excited element
     ana_greens = AnalyticalGreens(2, 2, nodes, elements, excitation_index)  
     greens = run_environment(elements, nodes, bound, excitation_index, ana_greens)
     plot3D(nodes[:,0], nodes[:,1], greens,'G('+str(excitation_index[0])+',r) FEM')
     plot3D(nodes[:,0], nodes[:,1], ana_greens,'G('+str(excitation_index[0])+',r) Series')
     return greens, ana_greens

#=============================================================================
    # MAIN CODE #
#=============================================================================

if __name__=="__main__":
  #solves for greens function everywhere due to one excitation, using FEM and series solution.
  #plots the solution and prints the interpolated values at the centers of each element.
  greens, ana_greens = test_small_square()     
  



  