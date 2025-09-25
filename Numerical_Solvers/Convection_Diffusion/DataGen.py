#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D wave equation via FFT 

u_tt = c^2 * (u_xx + u_yy)

on [-1, 1]x[-1, 1], t > 0 and Dirichlet BC u=0

Based on: http://people.bu.edu/andasari/courses/numericalpython/python.html
"""
# %%
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  
from tqdm import tqdm 
file_path = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/'

from convection_diffusion_2D_implicit_JAX import * 


Lx, Ly = 10.0, 10.0   # Domain size
T = 5.0               # Total time
nx, ny = 64, 64       # Grid resolution
nt = 100              # Number of time steps
cx, cy = 0.5, 1.0     # Convection velocities
D = 0.5              # Diffusion coefficient
aa, bb = 2.0, 3.0     # Position of Gaussian 
cc = 4.0             # Width of Gaussian

n_sims = 100

#Initialising the Solver
solver = ConvectionDiffusionImplicitSolver(Lx, Ly, T, nx, ny, nt, cx, cy, D, bc='periodic')
def u0_gaussian(X, Y, aa, bb, cc):
    return jnp.exp(-((X - aa)**2 + (Y - bb)**2) / cc)

# %%
def LHS_Sampling(N=10):
    #Simulation Data Built using LHS sampling
    from pyDOE import lhs
    
    lb = np.asarray([1.0, 1.0, 1.0]) #aa, bb, cc
    ub = np.asarray([5.0, 5.0, 5.0]) #aa, bb, cc
    
       
    param_lhs = lb + (ub-lb)*lhs(3, N)
    
    list_u = []
    
    for ii in tqdm(range(N)):
                t, u = solver.solve(u0_gaussian, param_lhs[ii, 0], param_lhs[ii, 1], param_lhs[ii, 2])
                list_u.append(u)
        
    ic = param_lhs
    u = np.asarray(list_u)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    np.savez(f'{file_path}ConvDiff_D_{D}_cx_{cx}_cy_{cy}.npz', x=x, y=y, t=t, u=u, ic=ic)
LHS_Sampling(n_sims)
# %%
