# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:25:59 2021

Matlab how to do Neumann for FEM
@author: gross
"""
clear;

n = 64;
lx = 1.0;
# BCA = 0; 
# BCA = -0.1533;
BCA = 0.0883;   # CHANGED TO MATCH VALUE OF NEUMANN soltuion

# problem setup
dx=lx/(n-1);
x=0:dx:1;

# set up A's to compute Laplacian on internal points
nx=n-2;
e=ones(nx-2,1);
# left Dirichelt, right Neumann
AD = - spdiags([1 -2 1; e*[1 -2 1]; 1 -1 1],-1:1,nx,nx)/dx^2;
# both sides Neumann
AN = - spdiags([1 -1 1; e*[1 -2 1]; 1 -1 1],-1:1,nx,nx)/dx^2;

# set up RHS for internal points
b=zeros([nx 1]);
b(1:nx)=4*sin(33/pi.*x(2:nx+1));
#b(1:nx)=-2*pi*cos(2*pi.*x(2:nx+1));

# incorporate left Dirichelt BC conditon into RHS
# e.g. p.2 of https://web.stanford.edu/class/cs205b/lectures/lecture16.pdf
bD=b-mean(b);  #CHANGED SO THAT RHS ENCODES COMPATIBILITY CONDITION!
bD(1)=-AD(2,1)*BCA; # fix xD(1) to random value BCA by adjusting rhs

# ensure Discrete Compatibility for Neumann case, i.e. mean(bN)=0
bN=b-mean(b);

# solve both with CG
xD=AD\bD; #pcg(AD,bD,1e-6,2*n);
xN=pcg(AN,bN,1e-6,2*n);

# solutions are obtained on internal points only and boundary conditions
# are incorporated into the RHS. For plotting values need to be appended.
xD=[ BCA ; xD; xD(end)];
xN=[xN(1); xN; xN(end)];

# compute first derivatives, frequently of interest for potential problems
# and also to see the Neumann boundary conditions
dxDdx=gradient(xD,dx);
dxNdx=gradient(xN,dx);

# plot solutions
f=figure(1); set(f,'Position', [200, 300, 720, 310]);
subplot(1,2,1);
plot(x,xD,'r-'); hold on
plot(x,xN,'b--'); hold off
title('Solutions'); grid on;

subplot(1,2,2);
plot(x,dxDdx,'r-'); hold on
plot(x,dxNdx,'b--'); hold off
title('First Derivatives of Solutions'); grid on;
# legend('left Dirichelt','both Neumann','location','SouthEast');
legend('left Dirichelt','both Neumann','location','NorthEast');