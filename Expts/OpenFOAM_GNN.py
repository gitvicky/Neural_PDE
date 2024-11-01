#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulent Cylinder Wake from OpenFOAM modelled using a GNN 

"""

# %%
configuration = {"Case": 'Turbulent Wake',
                 "Field": 'u, v, p',
                 "Model": 'GCN',
                 "Epochs": 0,
                 "Batch Size": 5,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'ReLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 1,    
                 "T_out": 20,
                 "Step": 1,
                 "Width": 64, 
                 "Variables":3, 
                 "Loss Function": 'MSE',
                 "UQ": 'None', #None, Dropout
                 }

import os
# from simvue import Run
# run = Run(mode='disabled')
# run.init(folder="/Neural_PDE", tags=['NPDE', configuration['Model'], 'AR', 'Wake', 'Turbulent'], metadata=configuration)

# #Saving the current run file and the git hash of the repo
# run.save_file(os.path.abspath(__file__), 'code')
# import git
# repo = git.Repo(search_parent_directories=True)
# sha = repo.head.object.hexsha
# run.update_metadata({'Git Hash': sha})

# %% #Importing the necessary packages
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from tqdm import tqdm 
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt

#Adding the NPDE package to the system python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Importing the models and utilities. 
from Neural_PDE.Models.GNNs import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import *
# %% 
#Loading the airfoil data
import os
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/simvue_testing/OpenFOAM/turbulent/' 
data_loc = '/Users/Vicky/GNN4PDE/Data/'
data_loc = data_loc + 'airfoil_turbulent_gnns.npz'
data = np.load(data_loc)

x, y, t = data['x'], data['y'], data['t']
u, v, p = data['u'], data['v'], data['p']
uu = np.concatenate((np.expand_dims(u, -1), np.expand_dims(v, -1), np.expand_dims(p, -1)), axis=-1)
xx = np.stack((x,y)).T

uu = torch.tensor(uu, dtype=torch.float32)
xx = torch.tensor(xx, dtype=torch.float32)

ntrain = 10
ntest = 5 

#Normalising the dataset with the preferred normalisation. 
norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    normalizer = MinMax_Normalizer
elif norm_strategy == 'Range':
    normalizer = RangeNormalizer
elif norm_strategy == 'Gaussian':
    normalizer = GaussianNormalizer

node_normalizer = normalizer(uu)
loc_normalizer = normalizer(xx)

uu_norm = node_normalizer.encode(uu)
xx_norm = loc_normalizer.encode(xx)

train_uu = uu_norm[:ntrain]
test_uu = uu_norm[-ntest:]

# %%
#Model Setup

if configuration['Model'] == 'GCN':
    model = GCN(in_channels=3, hidden_channels=64, out_channels=3, num_layers=6).to(device)
if configuration['Model'] == 'NNConv':
    model = NNConvNet(in_channels=3, hidden_channels=64, out_channels=3, num_layers=2, edge_dim=4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = nn.MSELoss()
# %% 
# Training 
epochs = configuration['Epochs']
for ep in tqdm(range(epochs)):
    model.train()
    loss_epoch = 0
    for ii in tqdm(range(len(t)-2)):
        graph_data = get_graph(train_uu[:, ii], xx_norm, r=0.1)
        graph_data.y = train_uu[:, ii+1]
        if configuration['Model'] == 'GCN':
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        if configuration['Model'] == 'NNConv':
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.edge_attr.to(device))
        loss = loss_func(out, graph_data.y.to(device))
        loss.backward()
        loss_epoch += loss.item()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {ep+1}/{epochs}, Loss: {loss_epoch/ntrain:.6f}')

# %% 
#Evaluation. 
with torch.no_grad():
    model.eval()
    loss_eval = 0
    pred_uu = []
    for ii in tqdm(range(len(t)-2)):
        graph_data = get_graph(test_uu[:, ii], xx_norm, r=0.1)
        graph_data.y = test_uu[:, ii+1]
        if configuration['Model'] == 'GCN':
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        if configuration['Model'] == 'NNConv':
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.edge_attr.to(device))        
        pred_uu.append(out)
        loss = loss_func(out, graph_data.y.to(device))
        loss_eval += loss.item()
    print(f'Eval Loss: {loss_eval/ntest:.6f}')

pred_uu = [t.unsqueeze(0) for t in pred_uu]
pred_uu = torch.cat(pred_uu, dim=1)
# %% 
#Plotting

from matplotlib.tri import Triangulation
def plot_unstructured(x, y, values, title="Field"):
    """
    Create a color plot of values on an unstructured grid.
    
    Parameters:
    x (array-like): x-coordinates of the grid points
    y (array-like): y-coordinates of the grid points
    values(array-like):values at each (x,y) point
    title (str): title for the plot
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create triangulation
    triang = Triangulation(x, y)
    
    # Create the color plot
    contour = ax.tripcolor(triang, values, 
                          shading='gouraud',  # For smooth color transitions
                          cmap='plasma',
                          edgecolors='k')     # Color scheme
    
    # Add a colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Magnitude')
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # Make the plot aspect ratio equal
    ax.set_aspect('equal')
    
    # Add grid
    # ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig, ax

# Create the plot
sim_idx = 0 
t_idx = 10
var = 0
fig, ax = plot_unstructured(x, y, test_uu[sim_idx, t_idx+1, : ,var])
fig, ax = plot_unstructured(x, y, pred_uu[sim_idx, t_idx, : ,var])
plt.show()
# plt.savefig('comparison.png')
# %%
