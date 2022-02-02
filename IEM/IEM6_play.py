
"""
Script: IEM6.py
Author: Esther Grossman
Date: 2/3/21

Solving with pulse basis functions. The start and goal points are ON the boundary.
Potential specified EVERYWHERE. Normal derivative calculated on all boundaries
and start/goal position.

"""

import ipdb
import numpy as np
import numpy.linalg as la
import math

pi = math.pi
ln = np.lib.scimath.log
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("dark_background")


def plot3D(x, y, z):
    """Plots anything in 3D. Intended for showing potential surface over 2D plane."""
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title("IEM grid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("$\phi$")
    plt.show()


def make_box(lseg, Nx, Ny, corner=[0, 0]):
    """Nx is the number of discretized point you want on the x dimension. 1 points on
    each boundary, this would be Nx=1. lseg is the length between points, so the total
    dimension along x would be lseg*Nx. corner is a coordinate(list) that tells where to
    put the lower left corner of the box"""
    x = corner[0]
    y = corner[1]
    box = []
    for i in range(Nx):
        box.append(
            [round(x, 2), round(y, 2)]
        )  # must round because imperfect adding of decimals
        x += lseg
    for i in range(Ny):
        box.append([round(x, 2), round(y, 2)])
        y += lseg
    for i in range(Nx):
        box.append([round(x, 2), round(y, 2)])
        x -= lseg
    for i in range(Ny):
        box.append([round(x, 2), round(y, 2)])
        y -= lseg
    return box


def list_avg(x1, x2):
    assert isinstance(x1, list)
    """take two points as lists and return their midpoint as a list"""
    diff = np.subtract(x2, x1) / 2
    avg = list(np.round(np.add(diff, x1), 2))
    return avg


def dist(x1, x2):
    """Distance between two points"""
    r = math.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)
    return r


def greens(x1, x2):
    """Input is two cartesian coordinates. Output in Green's potential"""
    U = 1 / (2 * math.pi) * (math.log(1 / dist(x1, x2)))
    return U


def find_normal(q1, q2):
    if np.round(q2[1], 3) == np.round(q1[1], 3):
        normal = np.array([0, 1])
        if q2[0] > q1[0]:
            normal *= -1
    else:
        m = -(q2[0] - q1[0]) / (q2[1] - q1[1])
        normal_temp = np.array([1, m])
        normal = normal_temp / np.linalg.norm(normal_temp)
    if q2[1] < q1[1]:
        normal *= -1
    return normal


def intgreens(p, q1, q2, is_selfterm = False):
    """Input is reference point p and the segment points to integrate Greens function over. Output is 
    integral of Green's potential at point p is the center of the self-term"""
    v = 0 if q1[1] == q2[1] else 1  # v is variable to integrate over
    w = 1 if q1[1] == q2[1] else 0  # w is other variable, held constant over segment
    Greens = (
        lambda x: -1
        / (2 * pi)
        * ln(np.lib.scimath.sqrt((p[v] - x) ** 2 + (p[w] - q1[w]) ** 2))
    )
    
    if is_selfterm:
        #print(p,q1,q2, is_selfterm)
        Gint = (
            -1
            / (2 * pi)
            * (
                -(q1[v] - p[v]) * ln(abs(q1[v] - p[v]))
                + (q2[v] - p[v]) * ln(abs(q2[v] - p[v]))
                + q1[v]
                - q2[v])) 
    elif q1[v] == p[v]:
        Gint = (
            -1
            / (2 * pi)
            * (
                1 / 2 * (q2[v] - p[v]) * ln((p[w] - q1[w]) ** 2 + (q2[v] - p[v]) ** 2)
                + (p[w] - q1[w]) * math.atan((q2[v] - p[v]) / (p[w] - q1[w]))
                - q2[v]
                + q1[v]
            )
        )
    elif q2[v] == p[v]:
        Gint = (
            -1
            / (2 * pi)
            * (
                -(
                    1
                    / 2
                    * (q1[v] - p[v])
                    * ln((p[w] - q1[w]) ** 2 + (q1[v] - p[v]) ** 2)
                    + (p[w] - q1[w]) * math.atan((q1[v] - p[v]) / (p[w] - q1[w]))
                    - q1[v]
                )
                - q2[v]
            )
        ) 
    elif (abs(q1[v] - p[v]) + abs(q2[v] - p[v]) <= abs(q2[v] - q1[v])):
        Gint = (
            -1
            / (2 * pi)
            * (
                -(
                    1
                    / 2
                    * (q1[v] - p[v])
                    * ln((p[w] - q1[w]) ** 2 + (q1[v] - p[v]) ** 2)
                    + (p[w] - q1[w]) * math.atan((q1[v] - p[v]) / (p[w] - q1[w]))
                    - q1[v]
                )
                + (
                    1 / 2 * (q2[v] - p[v]) * ln((p[w] - q1[w]) ** 2 + (q2[v] - p[v]) ** 2)
                    + (p[w] - q1[w]) * math.atan((q2[v] - p[v]) / (p[w] - q1[w]))
                    - q2[v]
                )
            )
        )
    else:
        Gint = scipy.integrate.quad(Greens, q1[v], q2[v])[0]
          
    return Gint


def intDgreens(p, q1, q2, is_selfterm=False):
    """Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of grad Green's potential at point p"""
    v = 0 if q1[1] == q2[1] else 1  # v is variable to integrate over
    w = 1 if q1[1] == q2[1] else 0  # w is other variable, held constant over segment
    exterior_normal_vector = find_normal(q1, q2)
    Dgreens = (
        lambda x: 1
        / (2 * pi)
        * (
            (p[w] - q1[w]) * exterior_normal_vector[w]
            + (p[v] - x) * exterior_normal_vector[v]
        )
        / ((p[w] - q1[w]) ** 2 + (p[v] - x) ** 2)
    )
    if is_selfterm:  # self-terms solution
        DGint = (
            scipy.integrate.quad(Dgreens, q1[v], p[v])[0]
            + scipy.integrate.quad(Dgreens, p[v], q2[v])[0]
        )
    else:
        DGint = scipy.integrate.quad(Dgreens, q1[v], q2[v])[0]
    return DGint


def intDgreensA(p, q1, q2, is_self):
    """Input is reference point p and the segment points to integrate derivative Greens function over. Output is 
    integral of Green's potential at point p"""
    v = 0 if q1[1] == q2[1] else 1  # v is variable to integrate over
    w = 1 if q1[1] == q2[1] else 0  # w is other varaible, held constant over segment
    Q = np.subtract(q2, q1)
    m = np.zeros(2)
    m[w] = 1
    # positive normal vector
    nw = -1 if np.cross(Q, m) > 0 else 1  # outer boundary normal pointing outward
    if p[w] == q1[w]:  # p and q on same border
        if round(abs(p[v] - q1[v]), 2) == round(abs(p[v] - q2[v]), 2):
            if q2[v] > q1[v]:
                return -1 / 2  # self terms
            else:
                return +1 / 2
        else:
            return 0  # border terms
    else:
        DUinta = -1 / (2 * pi) * math.atan((p[v] - q1[v]) / (p[w] - q1[w]))
        DUintb = -1 / (2 * pi) * math.atan((p[v] - q2[v]) / (p[w] - q2[w]))
    DUint = nw * (DUintb - DUinta)
    return DUint


# ============================================================================
# MAIN CODE #
# ============================================================================
# Set Parameters
intgreens.counter1 = 0
intgreens.counter2 = 0
intgreens.counter3 = 0
nx = 19  # SPECIFIC number of points on x (odd number)
ny = 19  # SPECIFIC number of points on y (odd number)
lseg = 2.0/nx  #specific length of each segment
# ipdb.set_trace()

# Discretize outer boundary
bounds = make_box(lseg, nx, ny)  # coordinates of discretized domain
boundsA = np.array(bounds)
N_total = len(boundsA)

# Set up boundary conditions for collocation
c_Q = np.full(N_total, 1 / 2)
p = [list_avg(bounds[i - 1], bounds[i]) for i in range(N_total)]  # collocation points
#p.append(p.pop(0))  #make sure p lines between correct bounds
goal_index = int(N_total / 3)
start_index = 7
u_Q = np.ones(N_total)  # known and unknown potential at boundary point p
u_Q[goal_index] = 0
gam_Q = np.zeros(N_total)
gam_Q[start_index] = 1
gam_Q[goal_index] = 1

# Set up matrix A (unknowns)
A = np.zeros((N_total, N_total))
for i in range(N_total):
    for k in range(N_total):
        is_self = True if i == k else False
        if k == start_index or k == goal_index:
            A[i, k] = intgreens(p[i], bounds[k - 1], bounds[k], is_self)
        else:
            A[i, k] = intDgreens(p[i], bounds[k - 1], bounds[k], is_self)
            if is_self:
                A[i, k] += c_Q[i]

# Set up matrix b (knowns)
b = np.zeros(N_total)
for i in range(N_total):
    for k in range(N_total):
        is_self = True if i == k else False
        if k == start_index or k == goal_index:
            b[i] += u_Q[k] * intDgreens(p[i], bounds[k - 1], bounds[k], is_self)
            if is_self:
                b[i] += u_Q[i] * c_Q[i]
        else:
            b[i] += gam_Q[k] * intgreens(p[i], bounds[k - 1], bounds[k], is_self)


# Solve matrix equation
coef = la.solve(A, b)  # coeff are all the values of gam
gam_Q[start_index], gam_Q[goal_index] = coef[start_index], coef[goal_index]
u_Q = coef
u_Q[start_index] = 1
u_Q[goal_index] = 0


# ===========================================================================
# Generate Plots #
# ===========================================================================

# plot estimated (reconstructed) potential on boundary, given the calculated gamma coefficients
fig = plt.figure()
ax = Axes3D(fig)
pA = np.array(p[0:N_total])
u_bound_est = []
for i in range(N_total):
    temp = 0
    for k in range(N_total):
        is_self = True if i == k else False
        temp += 2 * (
            gam_Q[k] * intgreens(pA[i], bounds[k - 1], bounds[k], is_self)
            - u_Q[k] * intDgreens(pA[i], bounds[k - 1], bounds[k], is_self)
        )
    u_bound_est.append(temp)
ax.plot(list(pA[:, 0]), list(pA[:, 1]), u_bound_est[0:N_total], color="blue")
plt.plot(boundsA[:, 0], boundsA[:, 1], "ro")


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(boundsA[:,0], boundsA[:,1], 'ro')
plt.plot(pA[:,0], pA[:,1], 'bo')
plt.axis('equal')
normals1 = np.zeros((N_total,2))
for i in range(N_total):
     normals1[i] = find_normal(bounds[i-1], bounds[i])
ax.quiver(pA[:,0],pA[:,1], normals1[:,0], normals1[:,1], color ='r') #U and V are the x and y components of the normal vectors
# ax.quiver(p_start[:,0],p_start[:,1], normals2[:,0], normals2[:,1], color ='r')
# ax.quiver(p_goal[:,0],p_goal[:,1], normals2[:,0], normals2[:,1], color ='r')

# =============================================================================
# # Plot ones on the boundary
# fig = plt.figure()
# ax = Axes3D(fig)
# pA = np.array(p[0:N_total])
# ax.plot(list(pA[:, 0]), list(pA[:, 1]), np.ones(N_total), color="blue")
# plt.plot(boundsA[:, 0], boundsA[:, 1], "ro")
# 
# =============================================================================

# =============================================================================
# # Solve for points inside the domain and plot potential inside domain
# X,Y = np.mgrid[0.1:lseg*nx-0.1:30j, 0.1:lseg*ny-0.1:30j]
# xy = np.vstack((X.flatten(), Y.flatten())).T #coordinates of interior as MNx2 vector
# phi = np.zeros((len(xy),1))
# for i in range(len(xy)):
#     for k in range(N_total):
#         phi[i] += 2*(gam_Q[k]*intgreens(xy[i],bounds[k-1],bounds[k])-u_Q[k]*intDgreens(xy[i], bounds[k-1], bounds[k]))
# plot3D(xy[:,0],xy[:,1], phi[:,0])
#
# =============================================================================


# condition number of matrix
# la.cond(A)
