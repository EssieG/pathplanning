# -*- coding: utf-8 -*-
"""
Author: Esther Grossman
Dste: 7/23/21
ECE 222C Project 2 : Find Greens Function for triangular elements on a domain
using Finite Element Methods. Phi = Greens

Type %matplotlib in the console first to produce plots you can rotate

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
    
def AnalyticalGreensEigen(a,b, nodes, elements, excitation_index):
    ''' get the eigenfunction expansion solution due to one excitation
    a,b    -> x,y boundaries of the rectangle    
    '''
    G=[]
    [n1, n2, n3] = elements[excitation_index[0]] #node numbers of element e
    [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
    x_prime = (v1[0]+v2[0]+v3[0])/3
    y_prime = (v1[1]+v2[1]+v3[1])/3  #coordinates of excitation (lies in the center of an element)
    for i in range(len(nodes)):
        summer= 0
        y = nodes[i,1]
        x = nodes[i,0]
        for m in range(0,100):
            for n in range(0,100):
                if m==0 and n==0:
                    cmn = 0
                    continue
                elif m==0:
                    cmn = 0.5
                elif n==0:
                    cmn = 0.5
                else:
                    cmn = 1
                summer += cmn * (1/((m*pi/a)**2+(n*pi/b)**2) * np.cos(n*pi*x_prime/a) * np.cos(m*pi*y_prime/b) * np.cos(n*pi*x/a) * np.cos(m*pi*y/b))
        G.append(summer * 4 / (a*b))
    
    return G, nodes

def AnalyticalGreens(a, b, nodes, elements, excitation_index):
    '''Define greens''' 
    #ipdb.set_trace()
    G=[]
    gm=[]
    [n1, n2, n3] = elements[excitation_index[0]] #node numbers of element e
    [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
    x, y = np.meshgrid( np.linspace(0,a,20), np.linspace(0,b,20) )
    xy = np.vstack((x.flatten(), y.flatten())).T
    x_prime = (v1[0]+v2[0]+v3[0])/3
    y_prime = (v1[1]+v2[1]+v3[1])/3  #coordinates of excitation (lies in the center of an element)
    for i in range(len(xy)):
        summer=0
        y = xy[i,1]
        x = xy[i,0]
        for m in range(1,100):
            W = np.cosh(m*pi/a*y_prime)*np.sinh(m*pi*(y_prime-b)/a) - np.sinh(m*pi/a*y_prime)*np.cosh(m*pi*(y_prime-b)/a)
            if y > y_prime:
                g_m = - 2/(m*pi)*np.cosh(m*pi/a*(y-b))*np.cosh(m*pi/a*y_prime)/W*np.cos(m*pi/a*x_prime)
            else:
                g_m= - 2/(m*pi)*np.cosh(m*pi/a*y)*np.cosh(m*pi/a*(y_prime-b))/W*np.cos(m*pi/a*x_prime)
            summer += g_m*np.cos(pi*m*x/a)    
        G.append(summer)
        
# =============================================================================
#     summer=0
#     y = xy[1,1]
#     x = xy[1,0]
#     for m in range(1,100):
#         W = np.cosh(m*pi/a*y_prime)*np.sinh(m*pi*(y_prime-b)/a) - np.sinh(m*pi/a*y_prime)*np.cosh(m*pi*(y_prime-b)/a)
#         if y > y_prime:
#             g_m = - 2/(m*pi)*np.cosh(m*pi/a*(y-b))*np.cosh(m*pi/a*y_prime)/W*np.cos(m*pi/a*x_prime)
#         else:
#             g_m = - 2/(m*pi)*np.cosh(m*pi/a*y)*np.cosh(m*pi/a*(y_prime-b))/W*np.cos(m*pi/a*x_prime)
#         gm.append(g_m)
#      
#     m = np.linspace(1,100,1)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(m, gm)
#     plt.show()
# =============================================================================

        
    return G, xy
# =============================================================================
# 
# def AnalyticalGreens_nodepoints(a, b, nodes, elements, excitation_index):
#     '''Define greens''' 
#     #ipdb.set_trace()
#     G=[]
#     [n1, n2, n3] = elements[excitation_index[0]] #node numbers of element e
#     [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
#     x_prime = (v1[0]+v2[0]+v3[0])/3
#     y_prime = (v1[1]+v2[1]+v3[1])/3  #coordinates of excitation (lies in the center of an element)
#     for i in range(len(nodes)):
#         summer=0
#         y = nodes[i,1]
#         x = nodes[i,0]
#         for m in range(1,100):
#             W = np.cosh(m*pi/a*y_prime)*np.sinh(m*pi*(y_prime-b)/a) - np.sinh(m*pi/a*y_prime)*np.cosh(m*pi*(y_prime-b)/a)
#             if y > y_prime:
#                 g_n = - 2/(m*pi)*np.cosh(m*pi/a*(y-b))*np.cosh(m*pi/a*y_prime)/W*np.cos(m*pi/a*x_prime)
#             else:
#                 g_n = - 2/(m*pi)*np.cosh(m*pi/a*y)*np.cosh(m*pi/a*(y_prime-b))/W*np.cos(m*pi/a*x_prime)
#             summer += g_n*np.cos(pi*m*x/a)    
#         G.append(summer)
#     return G, nodes
# =============================================================================

def AnalyticalGreens_nodepoints(a, b, nodes, elements, excitation_index):
    '''Define greens''' 
    #ipdb.set_trace()
    G=[]
    [n1, n2, n3] = elements[excitation_index[0]] #node numbers of element e
    [v1, v2, v3] = [nodes[n1-1],nodes[n2-1],nodes[n3-1]] #coords of each node
    x_prime = (v1[0]+v2[0]+v3[0])/3
    y_prime = (v1[1]+v2[1]+v3[1])/3  #coordinates of excitation (lies in the center of an element)
    for i in range(len(nodes)):
        summer = 0
        summer_old = 0
        y = nodes[i,1]
        x = nodes[i,0]
        
        #if abs(x_prime-x) > abs(y_prime-y):
        m=1
        while True:
            W = np.cosh(m*pi/a*y_prime)*np.sinh(m*pi*(y_prime-b)/a) - np.sinh(m*pi/a*y_prime)*np.cosh(m*pi*(y_prime-b)/a)
            if y > y_prime:
                g_n = - 2/(m*pi)*np.cosh(m*pi/a*(y-b))*np.cosh(m*pi/a*y_prime)/W*np.cos(m*pi/a*x_prime)
            else:
                g_n = - 2/(m*pi)*np.cosh(m*pi/a*y)*np.cosh(m*pi/a*(y_prime-b))/W*np.cos(m*pi/a*x_prime)
            summer += g_n*np.cos(pi*m*x/a)    

            if abs(summer_old-summer) < 0.001:
                break
            summer_old = summer
            m+=1
        G.append(summer)
    
    #Plot the convergence of gn to test solution
    ipdb.set_trace()
    plot_convergence(a,b,0.5,1.8, 0.75,1.75) #x, y, x',y'
                
    return G, nodes

def plot_convergence(a,b, x, y, x_prime, y_prime ):
    '''Plot g_n vs m for different values of y and yprime)'''
    gm = []
    m_s = np.linspace(1,100,100)
    for m in range(1,101):
        W = np.cosh(m*pi/a*y_prime)*np.sinh(m*pi*(y_prime-b)/a) - np.sinh(m*pi/a*y_prime)*np.cosh(m*pi*(y_prime-b)/a)
        if y > y_prime:
            g_m = - 2/(m*pi)*np.cosh(m*pi/a*(y-b))*np.cosh(m*pi/a*y_prime)/W*np.cos(m*pi/a*x_prime)
        else:
            g_m = - 2/(m*pi)*np.cosh(m*pi/a*y)*np.cosh(m*pi/a*(y_prime-b))/W*np.cos(m*pi/a*x_prime)
        gm.append(g_m)
    print(m_s)
    print(gm)
    plt.plot(m_s, gm)
    plt.show()
   

 
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
        f = 1   #excitation over element
        b = np.full(3, A/3 * f) 
        #b = np.full(3, f/A) wrong
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
     
   # ipdb.set_trace()
    ##Subtract average RHS from RHS so sum is 0 (DISCRETE COMPATIBILITY)
    b_global = b_global - np.average(b_global)
    ##Set ones known Greens node (dirichlet) (SPECIFYING CONSTANT)                      
    for n in bound:          
        K_global[n,n] = 1
        b_global[n] = 0.1             #specify node value
        for i in range(0,num_phi): 
            if i != n:
                K_global[n,i] = 0 
            if i not in bound:        #Preserve symmetry
                b_global[i] -= K_global[i,n]*b_global[n]  
                K_global[i,n] = 0
    
    print(la.det(K_global))
    
    # Solve matrix
    phi = la.solve(K_global,b_global)
    
    # Interpolate to get greens at the center of each node
    greens_at_centers = []
    for el in range(num_e):
        temp = interpolate_greens(phi, elements, nodes, el, element_area[el])
        greens_at_centers.append(temp)
        
    # Interpolate to get greens at the center of each node for analytical solution
    ana_greens_at_centers = []
    for el in range(num_e):
        temp = interpolate_greens(phi_ana, elements, nodes, el, element_area[el])
        ana_greens_at_centers.append(temp)
  
    #print(la.cond(K_global))
    
    return phi, greens_at_centers, ana_greens_at_centers

def determine_constants( sln1, sln2 ):
    constants1 = np.empty(len(sln1))
    constants2 = np.empty(len(sln2))
    for i in range(len(sln1)):
        constants1[i] = sln1[0][i] - sln1[i][0]
        constants2[i] = sln2[0][i] - sln2[i][0]
        sln1[i] = sln1[i] + constants1[i]
        sln2[i] = sln2[i] + constants2[i]
    return sln1, sln2  

#=============================================================================
    # ENVIRONMENTS #
#=============================================================================
def test_small_square():
     print('Running test on 9-point square...\n')
     elements = np.loadtxt(fname ='elements_test.txt',delimiter = ",",dtype=int)
     nodes = np.loadtxt(fname = 'node_coord_test.txt',delimiter = ",") + np.ones(2)  #lower left corner is origin.
     bound = np.array([3])                 #set index of a "dirichlet node" which will fix greens at a node
     greens_slns=[] #list of solutions. each is an array of the greens value for the centers of element due to excitation at the center of anthoer
     ana_greens_slns=[]
    
     for i in range(len(elements)):
         excitation_index = np.array([i])      #set index of excited element (whole triangle)
         ana_greens, _ = AnalyticalGreensEigen(2, 2, nodes, elements, excitation_index)  
         greens_vertices, greens_centers, ana_greens_centers = run_environment(elements, nodes, bound, excitation_index, ana_greens)
         greens_slns.append(greens_centers)
         ana_greens_slns.append(ana_greens_centers)
     #Plots for very last element excitation    
     plot3D(nodes[:,0], nodes[:,1], greens_vertices,'G('+str(excitation_index[0])+',r) FEM')
     plot3D(nodes[:,0], nodes[:,1], ana_greens,'G('+str(excitation_index[0])+',r) Series')
     
     return greens_slns, ana_greens_slns
     
def test_square():
     print('Running test on square...\n')
     elements = np.loadtxt(fname ='elements_box.txt', delimiter = ",",dtype=int)
     nodes = np.loadtxt(fname = 'node_coord_box.txt', delimiter = ",") #lower left corner is origin.
     bound = np.array([3])                  #set index of a "dirichlet node" which will fix greens at a node
     greens_slns=[] #list of solutions. each is an array of the greens value for the centers of element due to excitation at the center of anthoer
     ana_greens_slns=[]

     #for i in range(len(elements)):
     for i in range(3):
         print('processing node ', i)
         excitation_index = np.array([i])      #set index of excited element (whole triangle)
         ana_greens_vertices, _ = AnalyticalGreensEigen(2, 2, nodes, elements, excitation_index)  
         greens_vertices, greens_centers, ana_greens_centers = run_environment(elements, nodes, bound, excitation_index, ana_greens_vertices)
         greens_slns.append(greens_centers)
         ana_greens_slns.append(ana_greens_centers)
     
     plot3D(nodes[:,0], nodes[:,1], greens_vertices,'G('+str(excitation_index[0])+',r) FEM')
     plot3D(nodes[:,0], nodes[:,1], ana_greens_vertices,'G('+str(excitation_index[0])+',r) Series')
     return greens_slns, ana_greens_slns
    

#=============================================================================
    # MAIN CODE #
#=============================================================================

if __name__=="__main__":
  #solves for greens function everywhere due to one excitation, using FEM and series solution.
  greens_raw_slns, ana_greens_raw_slns = test_small_square()  
  #greens_raw_slns, ana_greens_raw_slns = test_square()
  
  #Computes the final field, once all the constants are normalized for. 
  #Each array in the list corresponds to the field due to a particular excitation.
  #greens_slns[1][2] is the interpolated field at the center of element 2 due to excitation at center of element 1.
  greens_slns, ana_greens_slns = determine_constants(greens_raw_slns, ana_greens_raw_slns)


#=============================================================================
    # Debugging #
#=============================================================================
#Check if matrix is symmetric 
#(K_global.transpose() == K_global).all()

# Impose Boundary Conditions
# =============================================================================
# #                              ***for box potential***
# bound=[(value - 1) for value in bounds if value not in goals]  # indices of boundaries at a 1st potential
# goal=np.array([goals - 1])    #boundaries at a 2nd potential
# bounds = bounds - 1 
# =============================================================================
# =============================================================================
# #                            ***for matlab circle***
# bound=[(value-1) for value in bounds if value not in goals] #the -1 is becauce indices came from matlab which doesnt start at 0
# goal=np.array([goals-1])  #goal boundaries
# bounds=bounds-1           #all boundaries
# ===========================================================================



