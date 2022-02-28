#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:41:25 2022

This is a class for storing the small square FEM. The class can e updated once 
we find a mesh maker in python. right now, just importing some from matlab.

@author: ubuntu
"""
import numpy as np
import numpy.linalg as la
import ipdb

class FEM_domain:
  def __init__(self, elements, nodes, boundary_type, start = None, goal = None):
    #Assign variables that are necessary for FEM. Create functions for all needed computations.
    self.start = start
    self.goal = goal
    self.elements = elements
    self.nodes = nodes
    self.centers = self.get_center_of_elements()
    self.areas = self.get_area()
    self.boundary_type = boundary_type  #dirichlet, neumann, mixed
    self.potential_field_at_nodes = None
    
    
  def get_area(self):
    element_area = []
    for i in self.elements:
        AB = self.nodes[i[1]-1]-self.nodes[i[0]-1] #vertex of point B - vertex of point A
        AC = self.nodes[i[2]-1]-self.nodes[i[0]-1] #vertex of point C - vertex of point A
        element_area.append(abs(np.cross(AB, AC))*1/2)
    return element_area

  def get_center_of_elements(self):
    '''still got to make it return center for all elements'''
    n1, n2, n3 = np.hsplit(self.elements - 1, 3)
    v1, v2, v3 = self.nodes[np.squeeze(n1)], self.nodes[np.squeeze(n2)], self.nodes[np.squeeze(n3)] 
    center_triangles = np.array([1/3*(v1[:,0] +v2[:,0]+v3[:,0]), 1/3*(v1[:,1] +v2[:,1]+v3[:,1])])
    return center_triangles
  
  def get_Ke(self, e, A):
    '''Input array of element indices, array of node coordinates, element number,
    and element area. Returns 3x3 Ke array for that element.'''
    assert isinstance(e,int)
    [n1, n2, n3] = self.elements[e] #node numbers of element e
    [v1, v2, v3] = [self.nodes[n1-1],self.nodes[n2-1],self.nodes[n3-1]] #coords of each node
    b=[v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]]
    c=[v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]]
    Ke=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Ke[i,j] = 1/(4*A)*(b[i]*b[j]+c[i]*c[j])
    return Ke

  def get_Be(self, e, A, excitation_element):
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
        
  def calculate_potential_field(self):
    ''' returns U at each of the nodes '''
    # Solve the elemental/local functional equation for every element
    num_e = len(self.elements)
    num_phi = len(self.nodes)
    K_local = np.zeros((num_e,3,3))
    b_local = np.zeros((num_e, 3))
    K_global = np.zeros((num_phi,num_phi))
    b_global = np.zeros(num_phi)
    
    #Create local K matrix
    for el in range(num_e):
        K_local[el,:,:] = self.get_Ke(el, self.areas[el])
    
    # Place local K into global matrix
    for el in range(num_e):
        for i in range(3):
            ni=self.elements[el,i]-1
            b_global[ni] += b_local[el,i]
            for j in range(3):
                nj=self.elements[el,j]-1
                K_global[ni,nj] += K_local[el,i,j]
     
    #Apply boundary conditions 
    if self.boundary_type == 'dirichlet':
        pass
    elif self.boundary_type == 'neumann':
        pass
    else: #do mixed (for base domain)  
        #Set dirichlet nodes    
        start_idx = self.find_closest_node(self.start)
        goal_idx = self.find_closest_node(self.goal)
        K_global[start_idx,start_idx]=1
        b_global[start_idx]=10        #value of potential at boundaries/for box is 0
        for i in range(0, num_phi):
            if i != start_idx:
                K_global[start_idx,i] = 0 
                b_global[i] -= K_global[i,start_idx]*b_global[start_idx]  
                K_global[i,start_idx] = 0    
        K_global[goal_idx,goal_idx]=1   #setting goal boundary condition
        b_global[goal_idx]=1 #value of potential at boundaries/for box is phi_0
        for i in range(0, num_phi):
            if i != goal_idx:
                K_global[goal_idx,i] = 0    
                if i != start_idx:
                    b_global[i] -= K_global[i,goal_idx]*b_global[goal_idx]
                    K_global[i,goal_idx] = 0 
                    
    # Solve matrix
    self.potential_field_at_nodes = la.solve(K_global,b_global)
    return 

  def get_greens_field_at_nodes(self, excitation_idx):
    ''' still messy'''
    # Solve the elemental/local functional equation for every element
    num_e = len(self.elements)
    num_phi = len(self.nodes)
    K_local = np.zeros((num_e,3,3))
    b_local = np.zeros((num_e, 3))
    K_global = np.zeros((num_phi,num_phi))
    b_global = np.zeros(num_phi)
    
    for el in range(num_e):
        K_local[el,:,:] = self.get_Ke(self.elements, self.nodes, el, self.areas[el])
        b_local[el,:] = self.get_Be(el, self.areas[el], excitation_idx)
    
    # Place local K into global matrix
    for el in range(num_e):
        for i in range(3):
            ni=self.elements[el,i]-1
            b_global[ni] += b_local[el,i]
            for j in range(3):
                nj=self.elements[el,j]-1
                K_global[ni,nj] += K_local[el,i,j]
     
    #Subtract average RHS from RHS so sum is 0 (DISCRETE COMPATIBILITY)
    b_global = b_global - np.average(b_global)
    #Set ones known Greens node (dirichlet) (SPECIFYING CONSTANT)       
    bound = [1] #random index
    for n in bound:          
        K_global[n,n] = 1
        b_global[n] = 0.1             #specify node value
        for i in range(0,num_phi): 
            if i != n:
                K_global[n,i] = 0 
            if i not in bound:        #Preserve symmetry
                b_global[i] -= K_global[i,n]*b_global[n]  
                K_global[i,n] = 0 
    #print(la.det(K_global))
   
    # Solve matrix
    phi = la.solve(K_global,b_global)
    return phi
 
  def interpolate_field_at_centers(self, el, sln_list):
    ''' once greens has been computed eerywhere, can interpolate for greens at a specific point'''
    [n1, n2, n3] = self.elements[el] #node numbers of element e
    [v1, v2, v3] = [self.nodes[n1-1],self.nodes[n2-1],self.nodes[n3-1]] #coords of each node
    center_triangle = self.centers(el)
    A = self.areas(el)
    b=np.array([v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]])
    c=np.array([v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]])
    a = np.array([v2[0]*v3[1]-v2[1]*v3[0], v3[0]*v1[1]-v3[1]*v1[0], v1[0]*v2[1]-v1[1]*v2[0]])
    Ne = 1/(2*A)*(a + b*center_triangle[0] + c*center_triangle[1])
    sln_interpolated = Ne[0]*sln_list[n1-1] + Ne[1]*sln_list[n2-1] + Ne[2]*sln_list[n3-1]
    return sln_interpolated    

  def find_rectangular_element(self, point):
    ipdb.set_trace()
    elements = self.elements - 1
    element_coord1 = self.nodes[elements[:,0]]
    element_coord2 = self.nodes[elements[:,1]]
    element_coord3 = self.nodes[elements[:,2]]
    element_xs = np.hstack((element_coord1[:,0,None], element_coord2[:,0,None], element_coord3[:,0,None]))
    element_ys = np.hstack((element_coord1[:,1,None], element_coord2[:,1,None], element_coord3[:,1,None]))
    validx = np.logical_and(np.min(element_xs, axis = 1) < point[0], point[0] < np.max(element_xs, axis = 1))
    validy = np.logical_and(np.min(element_ys, axis = 1) < point[1], point[1] < np.max(element_ys, axis = 1))
    valid = np.logical_and(validx, validy)
    element_index = np.argwhere(valid)
    return element_index

  def find_closest_node(self, point):
    ''' return the first node in the element where point lies '''
    element_idx = self.find_triangular_element(point)
    node_idx = self.elements[element_idx,0] - 1
    return node_idx
    

  def find_triangular_element(self, point):
    #Barycentric method, note there will be errors if point falls exactly on triangle edge. to solve use technique from here ;
    #http://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
    elements = self.elements - 1
    xy1 = self.nodes[elements[:,0]]
    xy2 = self.nodes[elements[:,1]]
    xy3 = self.nodes[elements[:,2]]
    a = ((xy2[:,1] - xy3[:,1])*(point[0] - xy3[:,0]) + (xy3[:,0] - xy2[:,0])*(point[1] - xy3[:,1])) / ((xy2[:,1] - xy3[:,1])*(xy1[:,0] - xy3[:,0]) + (xy3[:,0] - xy2[:,0])*(xy1[:,1] - xy3[:,1]))
    b = ((xy3[:,1] - xy1[:,1])*(point[0] - xy3[:,0]) + (xy1[:,0] - xy3[:,0])*(point[1] - xy3[:,1])) / ((xy2[:,1] - xy3[:,1])*(xy1[:,0] - xy3[:,0]) + (xy3[:,0] - xy2[:,0])*(xy1[:,1] - xy3[:,1]))
    c = 1 - a - b
    valida = np.logical_and(0 <= a, a <= 1)
    validb = np.logical_and(0 <= b, b <= 1)
    validc = np.logical_and(0 <= c, c <= 1)
    temp = np.logical_and(valida, validb)
    valid = np.logical_and(temp, validc)
    element_index = int(np.argwhere(valid))
    return element_index

    
  def Efield(self, point):
    '''Input a numpy array of the coordinates of the three vertices of the element.
    A is the area of the element. e_phi is a numpy array of the potentials at each
    of the three vertices. Returns the electric field vector as a numpy array [x,y].'''
    element_idx = self.find_triangular_element(point)
    [n1, n2, n3] = self.elements[element_idx] 
    [v1, v2, v3] = [self.nodes[n1-1],self.nodes[n2-1],self.nodes[n3-1]] 
    phi_at_nodes = np.array([self.potential_field_at_nodes[n1-1], self.potential_field_at_nodes[n2-1], self.potential_field_at_nodes[n3-1]])
    #a=[v2[0]*v3[1]-v2[1]*v3[0], v3[0]*v1[1]-v3[1]*v1[0], v1[0]*v2[1]-v1[1]*v2[0]] 
    b=np.array([v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]])
    c=np.array([v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]])
    E = -1/(2*self.areas[element_idx])*np.array([sum(b*phi_at_nodes),sum(c*phi_at_nodes)])   #x and y coordinates of E field
    return E

  def Ufield(self, point):
    #ipdb.set_trace()
    element_idx = self.find_triangular_element(point)
    [n1, n2, n3] = self.elements[element_idx] 
    [v1, v2, v3] = [self.nodes[n1-1],self.nodes[n2-1],self.nodes[n3-1]] 
    phi_at_nodes = np.array([self.potential_field_at_nodes[n1-1], self.potential_field_at_nodes[n2-1], self.potential_field_at_nodes[n3-1]])
    a=np.array([v2[0]*v3[1]-v2[1]*v3[0], v3[0]*v1[1]-v3[1]*v1[0], v1[0]*v2[1]-v1[1]*v2[0]]) 
    b=np.array([v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]])
    c=np.array([v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]])   
    #potential interpolation
    N=np.empty(3)
    for j in range(3):
        N[j]=1/(2*self.areas[element_idx])*(a[j]+b[j]*point[0]+c[j]*point[1])
    phi = sum(N*phi_at_nodes)
    return phi    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #get index of true, there should be only one. That is the element you ae in
