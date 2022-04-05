#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:41:25 2022

This is a class for storing the parameters of an integral equations object.

@author: ubuntu
"""
import numpy as np
import scipy.integrate
import numpy.linalg as la
pi = np.pi
sqrt = np.sqrt
ln = np.log
from meshEllipse import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ipdb

class IEM_anyshape:
  def __init__(self, center, radii, boundary_type, N = 30, start = None, goal = None, other = None, use_Greens = False, ob_list = []):
    ''' let boundary be another object of IEM_ellipse (base_domain)
    ob_list is a list of (center, radii) tuples
    '''
    self.movements = [np.copy(center)]
    self.center = center
    self.radii = radii     
    self.boundary_type = boundary_type  #dirichlet, neumann, mixed
    self.N_points = N                   #number of discretized points
    self.co_points = self.ellipse_collocation(self.N_points+1, self.center, self.radii)  #loc of collocation points
    self.points = self.ellipse_knots(self.co_points) #loc of points (p)
    #self.points_t = np.absolute(self.co_points_t[:-1] + self.co_points_t[1:])/2      #parameterization of points
    self.start = start
    self.goal = goal
    self.other = False
    if start is not None:  
        self.N_startgoal_points = 15  #check normals
        self.startgoal_radii = [0.1, 0.1]
        self.start_co_points = self.ellipse_collocation(self.N_startgoal_points+1, self.start, self.startgoal_radii)
        self.goal_co_points = self.ellipse_collocation(self.N_startgoal_points+1, self.goal, self.startgoal_radii)
        self.start_points = self.ellipse_knots(self.start_co_points)
        self.goal_points = self.ellipse_knots(self.goal_co_points) 
        self.ob_radii = []
        self.ob_center = []
        self.ob_co_points = []
        self.ob_co_points_t = []
        self.ob_points = []
        self.N_ob_points = 15
        for i in range(len(ob_list)):
            self.ob_radii.append(ob_list[i][1])
            self.ob_center.append(ob_list[i][0])
            temp_co = self.ellipse_collocation(self.N_ob_points+1, self.ob_center[i], self.ob_radii[i])
            self.ob_co_points.append(temp_co)
            self.ob_points.append(self.ellipse_knots(self.ob_co_points[i]))
        #self.startgoal_points_t = np.absolute(self.startgoal_co_points_t[:-1] + self.startgoal_co_points_t[1:])/2  
    self.u_on_bounds, self.gam_on_bounds = None, None
    if other is not None:
        self.other = True
        self.use_Greens = use_Greens
        self.bound_center = other.center
        self.bound_radii = other.radii     
        self.N_bound_points = other.N_points                  #number of discretized points
        self.bound_co_points = self.ellipse_collocation(self.N_bound_points+1, self.bound_center, self.bound_radii)  #loc of collocation points
        self.bound_points = self.ellipse_knots(self.bound_co_points) #loc of points (p)
    
  def __eq__(self, other):
    return (self.center == other.center).all() and (self.radii == other.radii).all()
 
    
  def get_boundary_points(self):
    return self.co_points
    

  def ellipse_collocation(self, N, center, radii):
    '''Get collocation points'''
    t = np.linspace(0, 2, N)
    x1 = np.cos(pi*t)*radii[0] + center[0]
    x2 = np.sin(pi*t)*radii[1] + center[1] 
    x = np.column_stack((x1,x2))
    return x


  def ellipse_knots(self, collocation_points):
    ''' get boundary points '''
    x = collocation_points[:-1] + 0.5*(collocation_points[1:] - collocation_points[:-1]) 
    return x


  def greens(self, p, p2):
    ''' returns the Greens function at a point, p != p2 
    '''
    Greens = -1/(2*pi) * ln(sqrt((p[0]-p2[0])**2+(p[1]-p2[1])**2))
    return Greens

  def Dgreens(self, p, p2, q1, q2):
    ''' returns the derivative of Greens function at a point, p != p2
    derivative is wrt p2
    '''
    exterior_normal = find_normal(q1,q2)
    DGreens = 1/(2*pi)/((p[1]-p2[1])**2+(p[0]-p2[0])**2)*((p[0]-p2[0])*exterior_normal[0] + (p[1]-p2[1])*exterior_normal[1]) 
    return DGreens


  def intgreens(self, p, b1, b2, is_self = False):
      '''returns the greens function integrated over a curve, estimated as a linear segment.
         p is the point where the impulse is. b1 and b2 are the boundaries of the curve being integrated over.
         The analytical expression for this integral over the line segment connecting two boundary points.
         CORRECT
      '''
      #Using parameterization, from t=0 to t=1. For self terms, need to integrate drom t=0 to t=0.5 and t=0.5 to t=1 separately, if p is the center of the line segment.
      if is_self:  #point at center of segment
          #t=1
          #integral_upper = ((1 - 2*t)* ln(1/4* (1 - 2 *t)**2 * (c**2 - 2* c* x + u**2 - 2* u* y + x**2 + y**2)) + 4 *t)/(8 *pi)
          integral_upper = ((-1)* ln(1/4 * (b2[0]**2 - 2* b2[0]* b1[0] + b2[1]**2 - 2* b2[1]* b1[1] + b1[0]**2 + b1[1]**2)) + 4)/(8 *pi)
          integral_lower = (ln(1/4* (b2[0]**2 - 2* b2[0]* b1[0] + b2[1]**2 - 2* b2[1]* b1[1] + b1[0]**2 + b1[1]**2)))/(8 *pi)
      else:
          t =1
          integral_upper = -(((p[0] * (b1[0] - b2[0]) + p[1] * (b1[1] - b2[1]) + b2[0] * b1[0] + b2[1] * b1[1] - b1[0]**2 - b1[1]**2) * ln(p[0]**2 - 2 * t *(
              p[0] * (b2[0] - b1[0]) + p[1] * (b2[1] - b1[1]) - b2[0] * b1[0] - b2[1]* b1[1] + b1[0]**2 + b1[1]**2) - 2 * p[0] * b1[0] + p[1]**2 - 2 * p[1] * b1[1] + t**2 *(
              b2[0]**2 - 2* b2[0]* b1[0] + b2[1]**2 - 2* b2[1]* b1[1] + b1[0]**2 + b1[1]**2) + b1[0]**2 + b1[1]**2))/(b2[0]**2 - 2* b2[0]* b1[0] + b2[1]**2 - 2* b2[1]* b1[1] + b1[0]**2 + b1[1]**2) + (
              2* (p[0] *(b1[1] - b2[1]) + p[1]* (b2[0] - b1[0]) - b2[0]* b1[1] + b2[1]* b1[0]) * np.arctan((p[0]* (b1[0] - b2[0]) - p[1]* b2[1] + p[1] *b1[1] + b2[0]**2 * t + b2[0] *(
              b1[0] - 2* t* b1[0]) + t* b2[1]**2 - 2* t* b2[1] *b1[1] + t *b1[0]**2 + t *b1[1]**2 + b2[1] * b1[1] - b1[0]**2 - b1[1]**2)/(p[0] * (b1[1] - b2[1]) + p[1]* (b2[0] - b1[0]) - b2[0]* b1[1] + b2[1]* b1[0])))/(
              b2[0]**2 - 2* b2[0] *b1[0] + b2[1]**2 - 2* b2[1] *b1[1] + b1[0]**2 + b1[1]**2) + t* ln((p[0] - b2[0]* t + (t - 1)* b1[0])**2 + (p[1] + t *(b1[1] - b2[1]) - b1[1])**2) - 2 *t)/(4*pi)
          #t = 0 ( terms taken out already)
          integral_lower = -(((p[0] * (b1[0] - b2[0]) + p[1] * (b1[1] - b2[1]) + b2[0] * b1[0] + b2[1] * b1[1] - b1[0]**2 - b1[1]**2) * ln(p[0]**2 - 2 * p[0] * b1[0] + p[1]**2 - 2 * p[1] * b1[1] + b1[0]**2 + b1[1]**2))/(
              b2[0]**2 - 2* b2[0]* b1[0] + b2[1]**2 - 2* b2[1]* b1[1] + b1[0]**2 + b1[1]**2) + (
              2* (p[0] *(b1[1] - b2[1]) + p[1]* (b2[0] - b1[0]) - b2[0]* b1[1] + b2[1]* b1[0]) * np.arctan((p[0]* (b1[0] - b2[0]) - p[1]* b2[1] + p[1] *b1[1] + b2[0] *(
              b1[0] ) + b2[1] * b1[1] - b1[0]**2 - b1[1]**2)/(p[0] *(b1[1] - b2[1]) + p[1]* (b2[0] - b1[0]) - b2[0]* b1[1] + b2[1]* b1[0])))/(
              b2[0]**2 - 2* b2[0] *b1[0] + b2[1]**2 - 2* b2[1] *b1[1] + b1[0]**2 + b1[1]**2))/(4*pi)           
      integral = integral_upper - integral_lower
      integral *= sqrt((b2[0]-b1[0])**2 + (b2[1]-b1[1])**2)
      return integral
 
  def intgreens_selfterms_vectorized(self, b1, b2):
      '''
      vector b1, b2.
      '''
      integral_upper = ((-1)* ln(1/4 * (b2[:,0]**2 - 2* b2[:,0]* b1[:,0] + b2[:,1]**2 - 2* b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2)) + 4)/(8 *pi)
      integral_lower = (ln(1/4* (b2[:,0]**2 - 2* b2[:,0]* b1[:,0] + b2[:,1]**2 - 2* b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2)))/(8 *pi)
      integral = integral_upper - integral_lower
      integral *= sqrt((b2[:,0]-b1[:,0])**2 + (b2[:,1]-b1[:,1])**2)
      return integral

  def intgreens_vectorized(self, p, b1, b2):
      '''returns the greens function integrated over a curve, estimated as a linear segment.
         p is the point where the impulse is. b1 and b2 are the boundaries of the curve being integrated over.
         The analytical expression for this integral over the line segment connecting two boundary points.
         CORRECT
      '''
      #Using parameterization, from t=0 to t=1.
      #non-selfterms
      t =1
      integral_upper = -(((p[0] * (b1[:,0] - b2[:,0]) + p[1] * (b1[:,1] - b2[:,1]) + b2[:,0] * b1[:,0] + b2[:,1] * b1[:,1] - b1[:,0]**2 - b1[:,1]**2) * ln(p[0]**2 - 2 * t *(
          p[0] * (b2[:,0] - b1[:,0]) + p[1] * (b2[:,1] - b1[:,1]) - b2[:,0] * b1[:,0] - b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2) - 2 * p[0] * b1[:,0] + p[1]**2 - 2 * p[1] * b1[:,1] + t**2 *(
          b2[:,0]**2 - 2* b2[:,0]* b1[:,0] + b2[:,1]**2 - 2* b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + b1[:,0]**2 + b1[:,1]**2))/(b2[:,0]**2 - 2* b2[:,0]* b1[:,0] + b2[:,1]**2 - 2* b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + (
          2* (p[0] *(b1[:,1] - b2[:,1]) + p[1]* (b2[:,0] - b1[:,0]) - b2[:,0]* b1[:,1] + b2[:,1]* b1[:,0]) * np.arctan((p[0]* (b1[:,0] - b2[:,0]) - p[1]* b2[:,1] + p[1] *b1[:,1] + b2[:,0]**2 * t + b2[:,0] *(
          b1[:,0] - 2* t* b1[:,0]) + t* b2[:,1]**2 - 2* t* b2[:,1] *b1[:,1] + t *b1[:,0]**2 + t *b1[:,1]**2 + b2[:,1] * b1[:,1] - b1[:,0]**2 - b1[:,1]**2)/(p[0] * (b1[:,1] - b2[:,1]) + p[1]* (b2[:,0] - b1[:,0]) - b2[:,0]* b1[:,1] + b2[:,1]* b1[:,0])))/(
          b2[:,0]**2 - 2* b2[:,0] *b1[:,0] + b2[:,1]**2 - 2* b2[:,1] *b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + t* ln((p[0] - b2[:,0]* t + (t - 1)* b1[:,0])**2 + (p[1] + t *(b1[:,1] - b2[:,1]) - b1[:,1])**2) - 2 *t)/(4*pi)   
      #t = 0 ( terms taken out already)
      integral_lower = -(((p[0] * (b1[:,0] - b2[:,0]) + p[1] * (b1[:,1] - b2[:,1]) + b2[:,0] * b1[:,0] + b2[:,1] * b1[:,1] - b1[:,0]**2 - b1[:,1]**2) * ln(p[0]**2 - 2 * p[0] * b1[:,0] + p[1]**2 - 2 * p[1] * b1[:,1] + b1[:,0]**2 + b1[:,1]**2))/(
          b2[:,0]**2 - 2* b2[:,0]* b1[:,0] + b2[:,1]**2 - 2* b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + (
          2* (p[0] *(b1[:,1] - b2[:,1]) + p[1]* (b2[:,0] - b1[:,0]) - b2[:,0]* b1[:,1] + b2[:,1]* b1[:,0]) * np.arctan((p[0]* (b1[:,0] - b2[:,0]) - p[1]* b2[:,1] + p[1] *b1[:,1] + b2[:,0] *(
          b1[:,0] ) + b2[:,1] * b1[:,1] - b1[:,0]**2 - b1[:,1]**2)/(p[0] *(b1[:,1] - b2[:,1]) + p[1]* (b2[:,0] - b1[:,0]) - b2[:,0]* b1[:,1] + b2[:,1]* b1[:,0])))/(
          b2[:,0]**2 - 2* b2[:,0] *b1[:,0] + b2[:,1]**2 - 2* b2[:,1] *b1[:,1] + b1[:,0]**2 + b1[:,1]**2))/(4*pi)           
      integral = integral_upper - integral_lower
      integral *= sqrt((b2[:,0]-b1[:,0])**2 + (b2[:,1]-b1[:,1])**2)
      return integral
      

  def intgreens_quad(self, p, b1, b2, is_selfterm = False):
    '''Input is reference point p and the segment points to integrate Greens function over. Output is 
    integral of Green's potential at point p
    x = b1[0]+ t* (b2[0]-b1[0])
    y = b1[1] + t* (b2[1]-b1[1])
    '''
    t1 = 0; t2 = 1
    Greens = lambda t : -1/(2*pi) * ln(sqrt((p[0]- (b1[0]+ t* (b2[0]-b1[0])))**2 + (p[1]-(b1[1] + t* (b2[1]-b1[1])))**2)) * sqrt((b2[0]-b1[0])**2+ (b2[1]-b1[1])**2)
    if is_selfterm:
        t_mid = (t1+t2)/2
        Gint = scipy.integrate.quad(Greens, t1, t_mid)[0] + scipy.integrate.quad(Greens, t_mid, t2)[0]
    else:
        Gint = scipy.integrate.quad(Greens, t1, t2)[0]
    return Gint


  def find_normal(self, q1, q2):
    if abs(q2[1] - q1[1]) <= 0.00001:
        normal = np.array([0,1])
    else:
        m = -(q2[0]-q1[0])/(q2[1]-q1[1])
        normal_temp = np.array([1, m])
        normal = normal_temp/np.linalg.norm(normal_temp)
    if q2[1] < q1[1]:
        normal *= -1
    return normal

  def find_normal_2D(self, q1, q2):
      '''it is assumed that q1,q2 goes counterclockwise around the boundary, 
      vectorized
      '''
      dx = q2[:,0] - q1[:,0] 
      dy = q2[:,1] - q1[:,1]
      vectors = np.vstack((dy, -dx)).T
      #ipdb.set_trace()
      normals = vectors/np.linalg.norm(vectors, axis=1)[:,None]
      #(-dy, dx) is the other, inward facing, normal
      return normals

  def intDgreens_quad(self, p, b1,b2, is_selfterm = False):
    '''Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p
    x = b1[0]+ t* (b2[0]-b1[0])
    y = b1[1] + t* (b2[1]-b1[1])
    '''
    t1 = 0; t2 = 1
    exterior_normal = self.find_normal(b1,b2)
    DGreens = lambda t : 1/(2*pi)/((p[1] - (b1[1] + t* (b2[1]-b1[1])))**2+(p[0] - (b1[0]+ t* (b2[0]-b1[0])))**2)*((p[0]-(b1[0]+ t* (b2[0]-b1[0])))*exterior_normal[0] + (p[1]-(b1[1] + t* (b2[1]-b1[1])))*exterior_normal[1]) * sqrt((b2[0]-b1[0])**2+ (b2[1]-b1[1])**2)
    if is_selfterm:
        t_mid = (t1+t2)/2
        DGint = scipy.integrate.quad(DGreens, t1, t_mid)[0] + scipy.integrate.quad(DGreens, t_mid, t2)[0]
    else:
        DGint = scipy.integrate.quad(DGreens, t1, t2, epsabs = 1.4*10**12)[0] 
    return DGint


  def intDgreens(self, p, b1, b2, is_selfterm = False):
    ''' Exact integration of 2-dimensional DGreens over a line segment.
        integral((q - (x + t(c - x))) n + (p - (y + t(u - y))) m)/((2 π) ((p - (y + t(u - y)))^2 + (q - (x + t(c - x)))^2)) dt 
        Calculates greens over a line segment. 
        DOESNT WORK FOR SELF TERMS - acoording to wolfram alpha, self terms do not converge.... principal value?
        integral((p - (x + t (c - x))) n + (q - (y + t (u - y))) m)/((2 π) ((q - (y + t (u - y)))^2 + (p - (x + t (c - x)))^2)) dt = ((-c n + m (y - u) + n x) log(t^2 (c^2 - 2 c x + u^2 - 2 u y + x^2 + y^2) - 2 t (c (p - x) - p x + q (u - y) - u y + x^2 + y^2) + p^2 - 2 p x + q^2 - 2 q y + x^2 + y^2) + 2 (c m - m x + n (y - u)) tan^(-1)((c^2 t + c (-p - 2 t x + x) + p x + q (y - u) + t u^2 - 2 t u y + t x^2 + t y^2 + u y - x^2 - y^2)/(c (q - y) + p (y - u) + x (u - q))))/(4 π (c^2 - 2 c x + u^2 - 2 u y + x^2 + y^2))
    '''
    exterior_normal = self.find_normal(b1,b2)
    #integrate from t = 0 to t= 1
    if is_selfterm:  #point at center of segment
        #Cauchy integral value, t = 0 to t= 0.5, function is antisymmetric, self terms go to zero
        #integral_principalValue = -(5.13068* (-c n + m (y - u) + n x))/(c^2 - 2 c x + u**2 - 2 *u *y + x**2 + y**2)
        integral = 0
    else:
        t = 1
        integral_upper = ((-b2[0] *exterior_normal[0] + exterior_normal[1] *(b1[1] - b2[1]) + exterior_normal[0]* b1[0]) *ln(
            t**2 *(b2[0]**2 - 2 *b2[0] *b1[0] + b2[1]**2 - 2* b2[1] *b1[1] + b1[0]**2 + b1[1]**2) - 2 *t *(
            b2[0] * (p[0] - b1[0]) - p[0]* b1[0] + p[1]* (b2[1] - b1[1]) - b2[1] *b1[1] + b1[0]**2 + b1[1]**2) + p[0]**2 - 2* p[0] *b1[0] + p[1]**2 - 2* p[1]*
            b1[1] + b1[0]**2 + b1[1]**2) + 2* (b2[0] *exterior_normal[1] - exterior_normal[1] *b1[0] + exterior_normal[0] *(b1[1] - b2[1])
            ) * np.arctan((b2[0]**2 *t + b2[0] *(-p[0] - 2 *t *b1[0] + b1[0]) + p[0]* b1[0] + p[1] *(b1[1] - b2[1]) +
            t * b2[1]**2 - 2 *t* b2[1] *b1[1] + t* b1[0]**2 + t* b1[1]**2 + b2[1]* b1[1] - b1[0]**2 - b1[1]**2)/(
            b2[0] * (p[1] - b1[1]) + p[0]* (b1[1] - b2[1]) + b1[0] *(b2[1] - p[1]))))/(4 * pi * (b2[0]**2 - 2* b2[0] *b1[0] + b2[1]**2 - 2 *b2[1]* b1[1] + b1[0]**2 + b1[1]**2)) 
         #t=0       
        integral_lower = ((-b2[0] *exterior_normal[0] + exterior_normal[1] *(b1[1] - b2[1]) + exterior_normal[0]* b1[0]) *ln(
                p[0]**2 - 2* p[0] *b1[0] + p[1]**2 - 2* p[1]* b1[1] + b1[0]**2 + b1[1]**2) + 2* (
                b2[0] *exterior_normal[1] - exterior_normal[1] *b1[0] + exterior_normal[0] *(b1[1] - b2[1])) * np.arctan(
                ( b2[0] *(-p[0] + b1[0]) + p[0]* b1[0] + p[1] *(b1[1] - b2[1]) + b2[1]* b1[1] - b1[0]**2 - b1[1]**2)/(
                b2[0] * (p[1] - b1[1]) + p[0]* (b1[1] - b2[1]) + b1[0] *(b2[1] - p[1]))))/(4 * pi * (b2[0]**2 - 2* b2[0] *b1[0] + b2[1]**2 - 2 *b2[1]* b1[1] + b1[0]**2 + b1[1]**2)) 
         
        integral = integral_upper - integral_lower
        integral *= sqrt((b2[0]-b1[0])**2 + (b2[1]-b1[1])**2)
    return integral


  def intDgreens_vectorized(self, p, b1, b2):
    ''' Exact integration of 2-dimensional DGreens over a line segment.
        integral((q - (x + t(c - x))) n + (p - (y + t(u - y))) m)/((2 π) ((p - (y + t(u - y)))^2 + (q - (x + t(c - x)))^2)) dt 
        Calculates greens over a line segment. 
        DOESNT WORK FOR SELF TERMS - acoording to wolfram alpha, self terms do not converge.... principal value?
        integral((p - (x + t (c - x))) n + (q - (y + t (u - y))) m)/((2 π) ((q - (y + t (u - y)))^2 + (p - (x + t (c - x)))^2)) dt = ((-c n + m (y - u) + n x) log(t^2 (c^2 - 2 c x + u^2 - 2 u y + x^2 + y^2) - 2 t (c (p - x) - p x + q (u - y) - u y + x^2 + y^2) + p^2 - 2 p x + q^2 - 2 q y + x^2 + y^2) + 2 (c m - m x + n (y - u)) tan^(-1)((c^2 t + c (-p - 2 t x + x) + p x + q (y - u) + t u^2 - 2 t u y + t x^2 + t y^2 + u y - x^2 - y^2)/(c (q - y) + p (y - u) + x (u - q))))/(4 π (c^2 - 2 c x + u^2 - 2 u y + x^2 + y^2))
    '''
    exterior_normal = self.find_normal_2D(b1,b2)
    #integrate from t = 0 to t= 1
    #if is_selfterm:  
    #    integral = np.zeros(len(b1))  ZEROOOOOS
    t = 1
    integral_upper = ((-b2[:,0] *exterior_normal[:,0] + exterior_normal[:,1] *(b1[:,1] - b2[:,1]) + exterior_normal[:,0]* b1[:,0]) *ln(
        t**2 *(b2[:,0]**2 - 2 *b2[:,0] *b1[:,0] + b2[:,1]**2 - 2* b2[:,1] *b1[:,1] + b1[:,0]**2 + b1[:,1]**2) - 2 *t *(
        b2[:,0] * (p[0] - b1[:,0]) - p[0]* b1[:,0] + p[1]* (b2[:,1] - b1[:,1]) - b2[:,1] *b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + p[0]**2 - 2* p[0] *b1[:,0] + p[1]**2 - 2* p[1]*
        b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + 2* (b2[:,0] *exterior_normal[:,1] - exterior_normal[:,1] *b1[:,0] + exterior_normal[:,0] *(b1[:,1] - b2[:,1])
        ) * np.arctan((b2[:,0]**2 *t + b2[:,0] *(-p[0] - 2 *t *b1[:,0] + b1[:,0]) + p[0]* b1[:,0] + p[1] *(b1[:,1] - b2[:,1]) +
        t * b2[:,1]**2 - 2 *t* b2[:,1] *b1[:,1] + t* b1[:,0]**2 + t* b1[:,1]**2 + b2[:,1]* b1[:,1] - b1[:,0]**2 - b1[:,1]**2)/(
        b2[:,0] * (p[1] - b1[:,1]) + p[0]* (b1[:,1] - b2[:,1]) + b1[:,0] *(b2[:,1] - p[1]))))/(4 * pi * (b2[:,0]**2 - 2* b2[:,0] *b1[:,0] + b2[:,1]**2 - 2 *b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2)) 
     #t=0       
    integral_lower = ((-b2[:,0] *exterior_normal[:,0] + exterior_normal[:,1] *(b1[:,1] - b2[:,1]) + exterior_normal[:,0]* b1[:,0]) *ln(
            p[0]**2 - 2* p[0] *b1[:,0] + p[1]**2 - 2* p[1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2) + 2* (
            b2[:,0] *exterior_normal[:,1] - exterior_normal[:,1] *b1[:,0] + exterior_normal[:,0] *(b1[:,1] - b2[:,1])) * np.arctan(
            ( b2[:,0] *(-p[0] + b1[:,0]) + p[0]* b1[:,0] + p[1] *(b1[:,1] - b2[:,1]) + b2[:,1]* b1[:,1] - b1[:,0]**2 - b1[:,1]**2)/(
            b2[:,0] * (p[1] - b1[:,1]) + p[0]* (b1[:,1] - b2[:,1]) + b1[:,0] *(b2[:,1] - p[1]))))/(4 * pi * (b2[:,0]**2 - 2* b2[:,0] *b1[:,0] + b2[:,1]**2 - 2 *b2[:,1]* b1[:,1] + b1[:,0]**2 + b1[:,1]**2)) 
     
    integral = integral_upper - integral_lower
    integral *= sqrt((b2[:,0]-b1[:,0])**2 + (b2[:,1]-b1[:,1])**2)
    return integral
    
    
  def calculate_potential_field(self):
    '''calculate the densities u and du/dn'''    
    if self.start is not None:
        N_obstacles = len(self.ob_center)
        N_total = self.N_points + 2 * self.N_startgoal_points + self.N_ob_points*N_obstacles
        p_all = np.vstack((self.points, self.start_points, self.goal_points))
        for ob in range(N_obstacles):
            p_all = np.vstack((p_all, self.ob_points[ob]))
        
        # Set up boundary conditions for collocation
        c_Q = np.full(N_total,1/2) 
        u_Q = np.ones(N_total)          #start potential is 1 and environment potential is unknown and obstacle potential is 1
        u_Q[N_total-self.N_ob_points*N_obstacles-2*self.N_startgoal_points:N_total-self.N_ob_points*N_obstacles-self.N_startgoal_points] = 1.1 
        u_Q[N_total-self.N_ob_points*N_obstacles-self.N_startgoal_points:N_total-self.N_ob_points*N_obstacles] = 0   #goal potential
        gam_Q = np.zeros(N_total)       #environment normal derivative is 0
        gam_Q[self.N_points : ] = 1     #unknown normal derivative on start and goal and obstacles
        u_indices = np.arange(self.N_points, N_total)    #indices where we know u
        gam_indices = np.arange(0,self.N_points)         #indices where we know gam
        
        #ipdb.set_trace()
        # Set up matrix A (unknowns) 
        A = np.zeros((N_total,N_total))
        #ipdb.set_trace()
        for i in range(N_total):
            for k in range(self.N_points):   #when integrating over environment segment
                is_self = True if i == k else False
                A[i,k] = self.intDgreens(p_all[i], self.co_points[k], self.co_points[k+1], is_self)
                if is_self:
                    A[i,k] += c_Q[i]
            for l in range(self.N_startgoal_points):    #integration over start segment
                is_self = True if i == self.N_points+l else False
                A[i,self.N_points+l] = self.intgreens(p_all[i], self.start_co_points[l], self.start_co_points[l+1], is_self)
            for h in range(self.N_startgoal_points):    #integration over goal segment
                is_self = True if i == self.N_points+self.N_startgoal_points+h else False
                A[i,self.N_points+self.N_startgoal_points+h] = self.intgreens(p_all[i], self.goal_co_points[h], self.goal_co_points[h+1], is_self)
            for o in range(N_obstacles):
                for m in range(self.N_ob_points):    #integration over goal segment
                    is_self = True if i == self.N_points+self.N_startgoal_points*2+self.N_ob_points*o+m else False
                    A[i,self.N_points+self.N_startgoal_points*2+self.N_ob_points*o+m] = self.intgreens(p_all[i], self.ob_co_points[o][m], self.ob_co_points[o][m+1], is_self)
        
        # Set up matrix b (knowns)
        b = np.zeros(N_total)
        for i in range(N_total):
            for k in range(self.N_points):
                is_self = True if i ==k else False
                b[i] += gam_Q[k] * self.intgreens(p_all[i], self.co_points[k], self.co_points[k+1], is_self)
            for l in range(self.N_startgoal_points):    #integration over start segment
                is_self = True if i == self.N_points+l else False
                b[i] += u_Q[self.N_points+l] * self.intDgreens(p_all[i], self.start_co_points[l], self.start_co_points[l+1], is_self)     
                if is_self:
                    b[i] += - u_Q[i]*c_Q[i]
            for h in range(self.N_startgoal_points):    #integration over goal segment
                is_self = True if i == self.N_points+self.N_startgoal_points+h else False                
                b[i] += u_Q[self.N_points + self.N_startgoal_points + h] * self.intDgreens(p_all[i], self.goal_co_points[h], self.goal_co_points[h+1], is_self)
                if is_self:
                    b[i] += - u_Q[i]*c_Q[i]      
            for o in range(N_obstacles):
                for h in range(self.N_ob_points):    #integration over goal segment
                    is_self = True if i == self.N_points + self.N_startgoal_points*2 + self.N_ob_points*o + h else False                
                    b[i] += u_Q[self.N_points + self.N_startgoal_points*2 + self.N_ob_points*o + h] * self.intDgreens(p_all[i], self.ob_co_points[o][h], self.ob_co_points[o][h+1], is_self)
                    if is_self:
                        b[i] += - u_Q[i]*c_Q[i]
                           
        # Solve matrix equation 
        coeff = la.solve(A,b) 
        gam_Q[u_indices] = coeff[u_indices]
        u_Q[gam_indices] = coeff[gam_indices]
    
    
    elif self.other == True:
        if self.use_Greens == True:
            # Set up boundary conditions for collocation
            N_total = self.N_points 
            c_Q = np.full(N_total,1/2)
            u_Q = np.ones(N_total)       #obstacles boundary set to 1
            p_all = self.points
            # Set up matrix A (unknowns)   
            A = np.zeros((N_total, N_total))
            b = np.zeros(N_total)
            b2 = np.zeros(N_total)
            #ipdb.set_trace()
            exterior_normal = self.find_normal_2D(self.co_points[:-1], self.co_points[1:])
            
            np.fill_diagonal(A, - self.intgreens_selfterms_vectorized(self.co_points[:-1], self.co_points[1:]))
            for i in range(N_total):
                not_self = np.ones(N_total, dtype=bool)
                not_self[i] = 0
                A[i,not_self] = - self.intgreens_vectorized(p_all[i], self.co_points[:-1][not_self], self.co_points[1:][not_self])
                b[i] += np.sum(- u_Q[not_self]*self.intDgreens_vectorized(p_all[i], self.co_points[:-1][not_self], self.co_points[1:][not_self]))
                for k in range(N_total): 
                    is_self = True if i == k else False
                    g_scatter, dg_gradient = self.Gfield(p_all[i], self.points[k])
                    A[i,k] += - g_scatter * get_segment_length(self.co_points[k], self.co_points[k+1])
                    # Set up matrix b (knowns)
                    dg_scatter = np.dot(dg_gradient, exterior_normal[k])
                    b[i] += - u_Q[k]*dg_scatter * get_segment_length(self.co_points[k], self.co_points[k+1]) 
                    if is_self: 
                        b[i] +=  u_Q[i]*c_Q[i]  
            #ipdb.set_trace()
                
        else:
            #ipdb.set_trace()
            # Set up boundary conditions for collocation
            N_total = self.N_points + self.N_bound_points 
            c_Q = np.full(N_total,1/2)   # 1/2 for environment
            u_Q = np.ones(N_total)       #obstacles boundary set to 1
            u_Q[0:self.N_bound_points] = 0 
            p_all =  np.vstack((self.bound_points, self.points))
            
            # Set up matrix A (unknowns)   
            A = np.zeros((N_total,N_total))
            for i in range(N_total):
                for k in range(self.N_bound_points): 
                    is_self = True if i == k else False
                    A[i,k] = self.intgreens(p_all[i], self.bound_co_points[k], self.bound_co_points[k+1], is_self)
                for l in range(self.N_points):
                    is_self = True if i == self.N_bound_points+l else False
                    A[i,self.N_bound_points+l] = - self.intgreens(p_all[i], self.co_points[l], self.co_points[l+1], is_self)
            
            # Set up matrix b (knowns)
            b = np.zeros(N_total)
            for i in range(N_total):
                for k in range(self.N_bound_points):
                    is_self = True if i == k else False
                    b[i] +=  u_Q[k]*self.intDgreens(p_all[i], self.bound_co_points[k], self.bound_co_points[k+1], is_self)
                    if is_self:
                        b[i] +=  u_Q[i]*c_Q[i]
                for l in range(self.N_points):
                    is_self = True if i == self.N_bound_points+l else False
                    b[i] += - u_Q[l+self.N_bound_points]*self.intDgreens(p_all[i], self.co_points[l], self.co_points[l+1], is_self)
                    if is_self:
                        b[i] += u_Q[i]*c_Q[i]
        
        # Solve matrix equation for gam on environment and obstacle
        gam_Q = la.solve(A,b) 
    
    self.u_on_bounds, self.gam_on_bounds = u_Q, gam_Q
    #self.plot_sample_points_inside()   #debugging
    #self.plot_normals()
    #self.plot_potential_on_boundaries(p_all)
    
    return


  def calculate_greens_densities(self):
    '''Use IEM to calculate the new greens field anywhere, due to a grid of excitations.
       Only for base domain objects.
       ONLY to be used with base domain objects
       
    '''
    xx, yy = np.meshgrid(np.arange(0, 2*self.radii[0]+1, 1.0), np.arange(0, 2*self.radii[1]+1, 1.0))
    excitations = np.vstack([xx.ravel(), yy.ravel()]).T
    valid_coords = np.empty(len(excitations), dtype = bool)
    for p in range(len(excitations)):
        valid_coords[p] = is_inside_polygon(self.points, excitations[p])
    #valid_coords = inside_base_ellipse(excitations[:,0], excitations[:,1], self.center, self.radii)
    excitations[~valid_coords,:] = np.nan
    g_on_bounds = []; dg_on_bounds= [];
    
    N_total = self.N_points 
    c_Q = np.full(N_total,1/2)    # 1/2 for environment
    u_Q = np.zeros(N_total)       # outer boundary set to zero
    p_all = self.points
    
    # Set up matrix A (unknowns)   
    A = np.zeros((N_total,N_total))
    for i in range(N_total):
        for l in range(self.N_points):
            is_self = True if i == l else False
            A[i,l] = self.intgreens(p_all[i], self.co_points[l], self.co_points[l+1], is_self)
    for xy_prime in excitations:  
        if np.isnan(xy_prime).any():
            g_on_bounds.append(np.nan) 
            dg_on_bounds.append(np.nan)
        else:
            # Set up matrix b (knowns)
            b = np.zeros(N_total)
            for i in range(N_total):
                b[i] = - self.greens(p_all[i], xy_prime)
                
            gam_Q = la.solve(A,b)
            #Save the densities 
            g_on_bounds.append(u_Q) 
            dg_on_bounds.append(gam_Q)
            
    return g_on_bounds, dg_on_bounds
    
          
  def Gfield(self, point1, point2):
        '''Used for planner 4. 
        Given a point, use the corresponding IEM equation to calculate the greens function there due to another point xy_prime'''
        N_env = self.N_bound_points
        c_env = self.bound_center
        r_env = self.bound_radii
        env = self.bound_co_points
        env_p = self.bound_points
        lbx, lby = np.floor(point2)
        ubx, uby = np.ceil(point2)
        if lbx == ubx:
            ubx += 1
        if lby == uby:
            uby += 1
        interpolation_points = [[lbx, lby], [lbx, uby], [ubx, lby], [ubx, uby]]
        interpolation_points_and_values = []
        #ipdb.set_trace()
        for xy_prime in interpolation_points:
            xy_prime_index = int(21*xy_prime[1] + xy_prime[0]) 
            dg = self.dg_on_bounds[xy_prime_index]
            #ipdb.set_trace()
            G_scat = np.sum(dg*self.intgreens_vectorized(point1, env[:-1], env[1:]))
# =============================================================================
#             G_scat=0
#             for k in range(N_env):
#                 G_scat += dg[k]*self.intgreens(point1, env[k], env[k+1])
#                 #G_scat += dg[k]*self.intgreens_estimate(point1, env[k], env[k+1])
#                 #G_scat += dg[k]*self.greens(point1, env_p[k]) * get_segment_length(env[k], env[k+1])
# =============================================================================
            interpolation_points_and_values.append((xy_prime[0], xy_prime[1], G_scat))   
        #print(interpolation_points_and_values)
        G_scatter = bilinear_interpolation(point2[0], point2[1], interpolation_points_and_values)
        DG_gradient = bilinear_gradient_interpolation(point2[0], point2[1], interpolation_points_and_values)
        #G_free = self.greens(point1, point2)
                
        return G_scatter, DG_gradient
   

  def Ufield(self, point):
    potential = 0
    if self.other == True:
        u = self.u_on_bounds
        gam = self.gam_on_bounds
        N_env = self.N_bound_points
        c_env = self.bound_center
        r_env = self.bound_radii
        env = self.bound_co_points
        N_ob = self.N_points
        c_ob = self.center
        r_ob = self.radii
        ob = self.co_points
        ob_p = self.points
        if self.use_Greens == True:
            #G, DG_grad = G_scattered_at_point(point, self.points, c_env, r_env)
            exterior_normal = self.find_normal_2D(ob[:-1], ob[1:])
            for l in range(N_ob):
                G_scatter, DG_grad_scatter = self.Gfield(point, self.points[l])
                DG_scatter = np.dot(DG_grad_scatter, exterior_normal[l])
                potential += - gam[l] * G_scatter*get_segment_length(ob[l], ob[l+1]) + u[l]*DG_scatter*get_segment_length(ob[l], ob[l+1]) 
            potential += np.sum(- gam * self.intgreens_vectorized(point, ob[:-1], ob[1:]) - u * self.intDgreens_vectorized(point, ob[:-1], ob[1:]))
                #potential += -(gam[l] * (G_scatter*get_arc_length(t_ob[l], t_ob[l+1], c_ob, r_ob) + self.intgreens_estimate(point, ob[l], ob[l+1])) - u[l]*( DG_scatter*get_arc_length(t_ob[l], t_ob[l+1], c_ob, r_ob) + self.intDgreens(point, t_ob[l], t_ob[l+1], c_ob, r_ob, ob[l], ob[l+1])))
                #potential += -(gam[l] * (G_scatter*get_arc_length(t_ob[l], t_ob[l+1], c_ob, r_ob) + self.greens(point, ob_p[l])*get_segment_length(ob[l], ob[l+1])) - u[l]*( DG_scatter*get_arc_length(t_ob[l], t_ob[l+1], c_ob, r_ob) + self.intDgreens(point, t_ob[l], t_ob[l+1], c_ob, r_ob, ob[l], ob[l+1])))
        else:
            for k in range(N_env):
                potential += (gam[k]*self.intgreens(point, env[k], env[k+1]) - u[k]*self.intDgreens(point, env[k], env[k+1]))
            for l in range(N_ob):
                potential += -(gam[N_env+l]*self.intgreens(point, ob[l], ob[l+1]) - u[N_env+l]*self.intDgreens(point, ob[l], ob[l+1]))
    else:
        u = self.u_on_bounds
        gam = self.gam_on_bounds
        N_env = self.N_points
        c_env = self.center
        r_env = self.radii
        env = self.co_points
        for k in range(N_env):
            potential += - (gam[k]*self.intgreens(point, env[k], env[k+1]) - u[k]*self.intDgreens(point, env[k], env[k+1]))

        if self.start is not None:
            N_sg = self.N_startgoal_points; N_ob = self.N_ob_points
            c_s = self.start; c_g = self.goal; c_ob = self.ob_center
            r_sg = self.startgoal_radii; r_ob = self.ob_radii
            start = self.start_co_points; ob = self.ob_co_points
            goal = self.goal_co_points
            potential *= -1  #normal points in opposite direction for env
            potential += np.sum(-gam[N_env : N_env+N_sg]*self.intgreens_vectorized(point, start[:-1], start[1:]))
            potential += np.sum(- gam[N_env+N_sg : N_env+2*N_sg]*self.intgreens_vectorized(point, goal[:-1], goal[1:]))
            for k in range(self.N_startgoal_points):
                potential += u[N_env+k]*self.intDgreens(point, start[k], start[k+1])
            for k in range(self.N_startgoal_points): 
                potential += u[N_env+N_sg+k]*self.intDgreens(point, goal[k], goal[k+1])
            for o in range(len(c_ob)):
                for k in range(self.N_ob_points): 
                   potential += - (gam[N_env+N_sg*2+o*N_ob+k]*self.intgreens(point, ob[o][k], ob[o][k+1]) - u[N_env+N_sg*2+o*N_ob+k]*self.intDgreens(point, ob[o][k], ob[o][k+1]))
    
    return potential   


  def Efield(self, point, dx = 0.1, dy = 0.1):
    '''Input a numpy array of the coordinates of the three vertices of the element.
    A is the area of the element. e_phi is a numpy array of the potentials at each
    of the three vertices. Returns the electric field vector as a numpy array [x,y].'''
    # Return x and y coordinates of E field
    U2 = self.Ufield(point + [dx, 0.0])
    U4 = self.Ufield(point - [dx, 0.0])
    U1 = self.Ufield(point + [0.0, dy])
    U3 = self.Ufield(point - [0.0, dy])                        
    E = - np.array( [(U2 - U4) / (2*dx), (U1 - U3) / (2*dy)] )
    return E


  def plot_potential_on_boundaries(self, p_all):
    # Plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    u_bound_est = [] 
    for i in range(len(p_all)):
        temp = 0
        if self.other ==True:
            for k in range(self.N_bound_points):
                is_self = True if i == k else False
                temp += 2*(self.gam_on_bounds[k]*self.intgreens(p_all[i], self.bound_co_points_t[k], self.bound_co_points_t[k+1], self.bound_center, self.bound_radii, is_self) - self.u_on_bounds[k]*self.intDgreens(p_all[i], self.bound_co_points_t[k], self.bound_co_points_t[k+1], self.bound_center, self.bound_radii, self.bound_co_points[k], self.bound_co_points[k+1], is_self))
            for k in range(self.N_points):
                is_self = True if i == self.N_bound_points+k else False
                temp += - 2*(self.gam_on_bounds[self.N_bound_points+k]*self.intgreens(p_all[i], self.co_points_t[k], self.co_points_t[k+1], self.center, self.radii, is_self) - self.u_on_bounds[self.N_bound_points+k]*self.intDgreens(p_all[i], self.co_points_t[k], self.co_points_t[k+1], self.center, self.radii, self.co_points[k], self.co_points[k+1], is_self))
        else:
            for k in range(self.N_points):
                is_self = True if i == k else False
                temp += - 2*(self.gam_on_bounds[k]*self.intgreens(p_all[i], self.co_points_t[k], self.co_points_t[k+1], self.center, self.radii, is_self) - self.u_on_bounds[k]*self.intDgreens(p_all[i], self.co_points_t[k], self.co_points_t[k+1], self.center, self.radii, self.co_points[k], self.co_points[k+1], is_self))
            if self.start is not None:
                temp *= -1 #for outer boundary normal is reversed
                for k in range(self.N_startgoal_points):
                    is_self = True if i == self.N_points+k else False
                    temp += - 2*(self.gam_on_bounds[self.N_points+k]*self.intgreens(p_all[i], self.startgoal_co_points_t[k], self.startgoal_co_points_t[k+1], self.start, self.startgoal_radii, is_self) - self.u_on_bounds[self.N_points+k]*self.intDgreens(p_all[i], self.startgoal_co_points_t[k], self.startgoal_co_points_t[k+1], self.start, self.startgoal_radii, self.start_co_points[k], self.start_co_points[k+1], is_self))
                for k in range(self.N_startgoal_points):
                    is_self = True if i == self.N_points+self.N_startgoal_points+k else False
                    temp += -2* (self.gam_on_bounds[self.N_points+self.N_startgoal_points+k]*self.intgreens(p_all[i], self.startgoal_co_points_t[k], self.startgoal_co_points_t[k+1], self.goal, self.startgoal_radii, is_self) - self.u_on_bounds[self.N_points+self.N_startgoal_points+k]*self.intDgreens(p_all[i], self.startgoal_co_points_t[k], self.startgoal_co_points_t[k+1], self.goal, self.startgoal_radii, self.goal_co_points[k], self.goal_co_points[k+1], is_self))
        u_bound_est.append(temp)
    #ax.plot(list(self.points[:,0]), list(self.points[:,1]), u_bound_est, color='blue')
    ax.plot(list(p_all[:,0]), list(p_all[:,1]), u_bound_est, color='blue')
    plt.plot(self.co_points[:,0], self.co_points[:,1], 'ro')
    plt.title('Boundary Potential')
    return


  def plot_normals(self): 
    # Plot collocation boundary & normals to verify
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.rcParams['font.size'] = '15'
    plt.plot(self.co_points[:,0], self.co_points[:,1], 'ro')
    plt.plot(self.points[:,0], self.points[:,1], 'bo')
    #plt.axis('equal')
    plt.title('Boundary Normals')
    normals1 = self.find_normal_2D(self.co_points[:-1], self.co_points[1:])
    if self.start is not None:
        normals2 = self.find_normal_2D(self.start_co_points[:-1], self.start_co_points[1:])
        ax.quiver(self.start_points[:,0],self.start_points[:,1], normals2[:,0], normals2[:,1], color ='r')
        ax.quiver(self.goal_points[:,0],self.goal_points[:,1], normals2[:,0], normals2[:,1], color ='r')  
        normals2 = np.zeros((self.N_ob_points,2))
        for o in range(len(self.ob_center)):
            for i in range(self.N_ob_points):
                normals2[i] = self.find_normal(self.ob_co_points[o][i], self.ob_co_points[o][i+1])
            ax.quiver(self.ob_points[o][:,0],self.ob_points[o][:,1], normals2[:,0], normals2[:,1], color ='g')
    ax.quiver(self.points[:,0],self.points[:,1], normals1[:,0], normals1[:,1], color ='r') #U and V are the x and y components of the normal vectors
    
    
  def plot_sample_points_inside(self):
      xy_inside = []
      if self.other == True:
          n = 3; r = np.array(self.radii)+[0.001,0.001]; c = self.center; ng = ellipse_grid_count(n, r, c)
          xy_inner = ellipse_grid_points(n, r, c, ng).T
          xy_outer = ellipse_grid_points(2*n, self.bound_radii-[0.1,0.1], self.bound_center, ellipse_grid_count(2*n, self.bound_radii-[0.001,0.001], self.bound_center)).T
          xy, in1 = outside_ellipse(xy_outer, c, r)
          xy_inside.append(in1)
          
      elif self.start is None:
          #Planner #3
          # Solve for points inside the domain and plot potential inside domain #NEEDS modifying
          n = 3; r = np.array(self.radii)+[0.001,0.001]; c = self.center; ng = ellipse_grid_count(n, r, c)
          xy_outer = ellipse_grid_points(2*n, self.radii-[0.001,0.001], self.center, ellipse_grid_count(2*n, self.radii-[0.001,0.001], self.center)).T
          xy, in1 = outside_ellipse(xy_outer, c, r)
          xy_inside.append(in1)
    
      else:
          n = 3; #r = np.array(self.startgoal_radii)+[0.001,0.001]; c = self.start; ng = ellipse_grid_count(n, r, c)
          #xy_start = ellipse_grid_points(n, r, c, ng).T
          #xy_goal = ellipse_grid_points(n, r, self.goal, ng).T
          xy_environment = ellipse_grid_points(2*n, np.array(self.radii)-[0.001,0.001], self.center, ellipse_grid_count(2*n, np.array(self.radii)-[0.001,0.001], self.center)).T
          xy, in1 = outside_ellipse(xy_environment, self.start, self.startgoal_radii)
          xy, in2 = outside_ellipse(xy, self.goal, self.startgoal_radii)
          xy_inside.extend((in1,in2))
          for o in range(len(self.ob_center)):
              xy, in4 = outside_ellipse(xy, self.ob_center[o], self.ob_radii[o])
              xy_inside.append(in4)
      phi = np.zeros((len(xy),1))
      for i in range(len(xy)):    
          phi[i] = self.Ufield(xy[i])
      xy_inside = np.vstack(xy_inside)
      xy = np.vstack((xy,xy_inside))
      phi = np.vstack((phi,np.ones((len(xy_inside),1))))
      plot3D(xy[:,0],xy[:,1], phi[:,0], title = 'Potential Field') 
      return
  
    
    
###################################Non-class functions###########################################
def plot3D(x,y,z, title = 'Potential'):
    '''Plots anything in 3D. Intended for showing potential surface over 2D plane.'''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.rcParams['font.size'] = '15'
    plt.title(title)
    #ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi$')
    #ax.set_zlabel(r'G ($\vecr$, (0,0)')
    plt.show()
    
def plot3D_overlap(x,y,z, x1,y1,z1, title = 'Potential Overlap'):
    '''Plots anything in 3D. Intended for showing potential surface over 2D plane.'''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.rcParams['font.size'] = '15'
    plt.title(title)
    #ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_trisurf(x1,y1,z1,cmap=cm.PiYG, linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi$')
    #ax.set_zlabel(r'G ($\vecr$, (0,0)')
    plt.show()

def outside_ellipse(pos, c, r):
    '''pos is list of positions Xx2. returns Xx2 array of positions that are
    outside the ellipse with center and radius c and r'''
    inequality = (pos[:,0]-c[0])**2/r[0]**2 + (pos[:,1]-c[1])**2/r[1]**2
    valid = pos[inequality > 1]; not_valid = pos[inequality <= 1]
    return valid, not_valid

def G_scattered_at_points(points, base_center, base_radii):
    '''pass a list of points and calculate G_scattered due to an excitation
    xy_excitation. 
    returns: N_o x N_o matrix of G_scatter where G_scatter(ij) is the greens 
    scatter function for obstacle boundary point i due to j
    '''
    points = points - base_center
    a = base_radii[0]
    x, y = points[:,0], points[:,1]
    g_scatter = np.full((points.shape[0], points.shape[0]), 0.0)
    dg_gradient_polar = np.empty((points.shape[0], points.shape[0],2))
    dg_gradient_cart = np.empty((points.shape[0], points.shape[0], 2))
    r,theta = cart2pol(x,y)
    
    for i in range(len(points)):
        THETA = theta[i]; R = r[i]
        g_scatter[:,i] = - (1/(2*np.pi) * np.log(a) - 1/(4*np.pi)*np.log(a**4 + R**2*r**2 - 2*a**2*R*r*np.cos(theta-THETA)))
        dg_gradient_polar[:,i,0] = 1/(4*np.pi) * 1/(a**4 + R**2*r**2 - 2*a**2*R*r*np.cos(theta-THETA)) * (2*R**2*r - 2*a**2*R*np.cos(theta-THETA))  #r hat direction
        dg_gradient_polar[:,i,1] = 1/(4*np.pi) * 1/(a**4 + R**2*r**2 - 2*a**2*R*r*np.cos(theta-THETA)) * ( 2*a**2*R*np.sin(theta-THETA))            #theta hat direction
        dg_gradient_cart[:,i, 0], dg_gradient_cart[:,i, 1] = pol2cart(dg_gradient_polar[:,i,0], dg_gradient_polar[:,i,1])
     
    #plot3D(x, y, g_scatter[:,7], title = 'Scattering at the obstacle due to the boundary')  
    
    return g_scatter, dg_gradient_cart


def G_scattered_at_point(point, excitations, base_center, base_radii):
    '''pass a list of points and calculate G_scattered due to an excitation
    xy_excitation. 
    returns: N_o x N_o matrix of G_scatter where G_scatter(ij) is the greens 
    scatter function for obstacle boundary point i due to j
    '''
    point = point - base_center
    excitations = excitations - base_center
    a = base_radii[0]
    g_scatter = np.empty(len(excitations))
    dg_gradient_polar = np.empty((excitations.shape[0],2))
    dg_gradient_cart = np.empty((excitations.shape[0], 2))
    r, theta = cart2pol(point[0],point[1])
    r2,theta2 = cart2pol(excitations[:,0],excitations[:,1])
    
    THETA = theta2; R = r2
    g_scatter[:] = - (1/(2*np.pi) * np.log(a) - 1/(4*np.pi)*np.log(a**4 + R**2*r**2 - 2*a**2*R*r*np.cos(theta-THETA)))
    dg_gradient_polar[:,0] = 1/(4*np.pi) * 1/(a**4 + R**2*r**2 - 2*a**2*R*r*np.cos(theta-THETA)) * (2*R**2*r - 2*a**2*R*np.cos(theta-THETA))  #r hat direction
    dg_gradient_polar[:,1] = 1/(4*np.pi) * 1/(a**4 + R**2*r**2 - 2*a**2*R*r*np.cos(theta-THETA)) * ( 2*a**2*R*np.sin(theta-THETA))            #theta hat direction
    dg_gradient_cart[:, 0], dg_gradient_cart[:, 1] = pol2cart(dg_gradient_polar[:,0], dg_gradient_polar[:,1])
     
    #plot3D(x, y, g_scatter[:,0], title = 'Scattering at the obstacle due to the boundary')  
    
    return g_scatter, dg_gradient_cart
  
    
def test_G_scattered(ob_domain):
    a = 10
    center = [0,0]
    xx, yy = np.meshgrid(np.arange(-a,a, 1), np.arange(-a,a,1))
    valid_coords = inside_base_ellipse(xx, yy, center, [a,a])
    
    g_scatter_ob, _ = G_scattered_at_points(ob_domain.points, ob_domain.bound_center, ob_domain.bound_radii)
    
    g_free = np.full((xx.shape[0], xx.shape[1], xx.size), 0.0)   #Nx X Ny X Npoints
    g_scatter = np.full((xx.shape[0], xx.shape[1], xx.size), 0.0)
    
    r, theta = cart2pol(xx,yy)
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
    plot3D(xx[valid_coords], yy[valid_coords], g_free[:,:,228][valid_coords], title = 'Free G')
    plot3D(xx[valid_coords], yy[valid_coords], g_scatter[:,:,228][valid_coords], title = 'Scattered G')
    plot3D(xx[valid_coords], yy[valid_coords], greens_table[:,:,228][valid_coords], title = 'Total G')
    
    plot3D_overlap(xx[valid_coords], yy[valid_coords], g_scatter[:,:,228][valid_coords],ob_domain.points[:,0],ob_domain.points[:,1],g_scatter_ob[:,7], title = 'overlap')
    return
    

def inside_base_ellipse(xx, yy, c, r):
    '''pos is list of positions Xx2. returns Xx2 array of positions that are
    outside the ellipse with center and radius c and r'''
    inequality = (xx-c[0])**2/r[0]**2 + (yy-c[1])**2/r[1]**2
    is_inside = inequality < 1
    return is_inside
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def get_segment_length(p1, p2):
    return np.linalg.norm(p1 - p2)


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

def bilinear_gradient_interpolation(x, y, points):
    '''Interpolate (dG/dx,dG/dy) from values associated with four points.
    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.
    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return np.array([ ((q22 - q12) * (y - y1) + (q21 - q11) * (y2 - y)) / ((x2 - x1) * (y2 - y1)) ,  
                      ((q12 - q11) * (x - x1) + (q22 - q21) * (x2 - x)) / ((x2 - x1) * (y2 - y1)) ])

def is_inside_convex_polygon(points, pointA, pointB):
    ''' given a point and a list of line segments, return true if the point is to the left of every line segment
    This is used to lay a grid of points over the base_domain.
    pointA is a Nx2 array
    point B is a Nx2 array
    point is the position you are evaluating if it is inside the shape
    NOT FINISHED
    '''
    dx = pointA[:,0] - pointB[:,0] 
    dy = pointA[:,1] - pointB[:,1]
    vectors = np.vstack((dy, -dx)).T
    return
    

# A Python3 program to check if a given point
# lies inside a given polygon
# Refer https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# for explanation of functions onSegment(),
# orientation() and doIntersect()
# Define Infinite (Using INT_MAX
# caused overflow problems)
# Given three collinear points p, q, r,
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
    if ((q[0] <= max(p[0], r[0])) &
		(q[0] >= min(p[0], r[0])) &
		(q[1] <= max(p[1], r[1])) &
		(q[1] >= min(p[1], r[1]))):
        return True	
    return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
	val = (((q[1] - p[1]) *
			(r[0] - q[0])) -
		((q[0] - p[0]) *
			(r[1] - q[1])))		
	if val == 0:
		return 0
	if val > 0:
		return 1 # Collinear
	else:
		return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):
	# Find the four orientations needed for
	# general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if (o1 != o2) and (o3 != o4):
		return True
	
	# Special Cases
	# p1, q1 and p2 are collinear and
	# p2 lies on segment p1q1
	if (o1 == 0) and (onSegment(p1, p2, q1)):
		return True

	# p1, q1 and p2 are collinear and
	# q2 lies on segment p1q1
	if (o2 == 0) and (onSegment(p1, q2, q1)):
		return True

	# p2, q2 and p1 are collinear and
	# p1 lies on segment p2q2
	if (o3 == 0) and (onSegment(p2, p1, q2)):
		return True

	# p2, q2 and q1 are collinear and
	# q1 lies on segment p2q2
	if (o4 == 0) and (onSegment(p2, q1, q2)):
		return True

	return False


def is_inside_polygon(points:list, p:tuple) -> bool:
    ''' return if point p is inside points '''
    n = len(points)
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
		
	# Create a point for line segment
	# from p to infinite
    INT_MAX = 1000
    extreme = (INT_MAX, p[1])
    count = i = 0
	
    while True:
        next = (i + 1) % n
		# Check if the line segment from 'p' to
		# 'extreme' intersects with the line
		# segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i], points[next], p, extreme)):				
			# If the point 'p' is collinear with line
			# segment 'i-next', then check if it lies
			# on segment. If it lies, return true, otherwise false
            if orientation(points[i], p, points[next]) == 0:
                return False #onSegment(points[i], p, points[next])									
            count += 1
        i = next
        if (i == 0):
            break	
        
	# Return true if count is odd, false otherwise
    return (count % 2 == 1)

# =============================================================================
# # =============================================================================
# # Test is_inside_polygon
# if __name__ == '__main__':
#     polygon1 = [ (0, 0), (10, 0), (10, 10), (0, 10) ]
#     p = (9, 8)
#     if (is_inside_polygon(points = polygon1, p = p)):
#         print ('Yes')
#     else:
#         print ('No')
# =============================================================================

# =============================================================================
# # Test is_inside_polygon
# if __name__ == '__main__':
#     polygon1 = np.array([ [0, 0], [10, 0], [10, 10], [0, 10] ])
#     p = np.array([10, 8])
#     if (is_inside_polygon(points = polygon1, p = p)):
#         print ('Yes')
#     else:
#         print ('No')
#     
# =============================================================================
    
    
    
    