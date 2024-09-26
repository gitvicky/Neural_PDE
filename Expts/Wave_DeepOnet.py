#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over the 2D Wave Equation auto-regressively 

Equation: u_tt = D*(u_xx + u_yy), D=1.0

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Model": 'DeepOnet',
                 "Epochs": 5000,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "Layers": 3,
                 "Width": 64, 
                 "Variables":1, 
                 "T_out": 20, 
                 "Loss Function": 'MSE',
                 "UQ": 'None', #None, Dropout
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'DeepONet', 'Wave'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import torch
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

#Adding the NPDE package to the system python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# %%
#Importing the models and utilities. 
from Neural_PDE.Models.DeepOnet import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Settung up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# Loading Data 
################################################################

t1 = default_timer()
data =  np.load(data_loc + '/Spectral_Wave_data_LHS.npz')

u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)[:configuration['T_out']]
u = torch.from_numpy(u_sol)[:, :configuration['T_out']][:100]
# u = u.permute(0, 2, 3, 1)

# %% 
#Setting up the Data for DeepOnet
#Class that takes in the simulation solutions, X Mesh and the initial (sensor) locations and gives you a dataset
from torch.utils.data import Dataset, DataLoader
class DON_Dataset(Dataset):
    def __init__(self, u, initial_locations):
        self.u = u
        self.initial_locations = initial_locations

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        u_sample = self.u[idx]
        u_ic = u_sample[0].flatten()
        u_init= u_ic[self.initial_locations]
        
        return torch.FloatTensor(u_init), torch.FloatTensor(u_sample)

#%% 
# Setting up the Data for DON
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.float32)
X,Y = torch.meshgrid(x,y, indexing='ij')
XY_loc = torch.column_stack((X.flatten(), Y.flatten()))
initial_locations = np.arange(0, len(x)*len(y))

# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    normalizer = MinMax_Normalizer
elif norm_strategy == 'Range':
    normalizer = RangeNormalizer
elif norm_strategy == 'Gaussian':
    normalizer = GaussianNormalizer
elif norm_strategy == 'Identity':
    normalizer = Identity

normalizer = normalizer(u)


# #Saving Normalisation 
# saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '_' + 'norms.npz'

# np.savez(saved_normalisations, 
#         a= normalizer.a.numpy(), b = normalizer.b.numpy()
# )

# run.save_file(saved_normalisations, 'output')

# %% 
#Setting up the Datasets and the Data Loaders
dataset = DON_Dataset(normalizer.encode(u), initial_locations)

train_size = ntrain = int(0.8 * len(dataset))
val_size = nval = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=configuration['Batch Size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=configuration['Batch Size'])

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################

model = DeepONet(in_branch=len(x)*len(y),
        width_branch=configuration['Width'],
        layers_branch=configuration['Layers'], 
        out_branch=configuration['Width'],
        in_trunk=3,
        width_trunk=configuration['Width'],
        layers_trunk=configuration['Layers'], 
        out_trunk=configuration['Width'])

model.to(device)

# run.update_metadata({'Number of Params': int(model.count_params())})
# print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

# %%
####################################
#Training Loop 
####################################


def train_one_epoch_don(model, train_loader, test_loader, loss_func, optimizer):
    model.train()
    train_loss = 0
    for u0, uu in train_loader:
        br = u0.unsqueeze(1).to(device)
        for tt in range(len(t)):
            tr = torch.FloatTensor(torch.column_stack((XY_loc, torch.ones(X.flatten().shape)*tt))).to(device)
            tr = tr.unsqueeze(0).repeat(br.shape[0], 1, 1)
            u_t = uu[:,tt].flatten(start_dim=1, end_dim=-1).to(device)
            
            optimizer.zero_grad()
            im = model(br, tr)
            # print(br.shape, tr.shape, u_t.shape, im.shape)
            loss = loss_func(im, u_t)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
            optimizer.step()
            train_loss += loss.item()

    train_loss =  train_loss / (len(train_loader)*train_loader.batch_size)
    
    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for u0, uu in test_loader:
            br = u0.unsqueeze(1).to(device)
            for tt in range(len(t)):
                tr = torch.FloatTensor(torch.column_stack((XY_loc, torch.ones(X.flatten().shape)*tt))).to(device)
                tr = tr.unsqueeze(0).repeat(br.shape[0], 1, 1)
                u_t = uu[:,tt].flatten(start_dim=1, end_dim=-1).to(device)
                im = model(br, tr)
                test_loss += loss_func(im, u_t)
        
        test_loss =  train_loss / (len(test_loader)*test_loader.batch_size)

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.


def validation_don(model, u0, uu):
    with torch.no_grad():
        pred = torch.zeros(uu.shape)
        br = u0.unsqueeze(1).to(device)
        for tt in range(len(t)):
            tr = torch.FloatTensor(torch.column_stack((XY_loc, torch.ones(X.flatten().shape)*tt))).to(device)
            tr = tr.unsqueeze(0).repeat(br.shape[0], 1, 1)
            u_t = uu[:,tt].flatten(start_dim=1, end_dim=-1).to(device)
            im = model(br, tr)
            pred[:,tt] = im.reshape(im.shape[0], len(x), len(y))
            
        # Performance Metrics
        MSE_error = (uu - pred).pow(2).mean()
        MAE_error = torch.abs(uu - pred).mean()

    return pred, MSE_error, MAE_error


start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch_don(model, train_loader, test_loader, loss_func, optimizer)
    t2 = default_timer()

    train_loss = train_loss  #/ ntrain / num_vars
    test_loss = test_loss #/ ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time

#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')
# %%
#Validation

#Testing 
test_u0, test_u_encoded = next(iter(test_loader))
pred_set_encoded, mse, mae = validation_don(model, test_u0, test_u_encoded)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#Denormalising and reshaping the target and the predictions
pred_set = normalizer.decode(pred_set_encoded.to(device)).cpu()
pred_set = pred_set.reshape(len(pred_set), len(t), len(x), len(y))

test_u = normalizer.decode(test_u_encoded.to(device)).cpu()
test_u = test_u.reshape(len(test_u), len(t), len(x), len(y))

# %% 
#Plotting performance

idx = 0
T_out = configuration['T_out']
u_field = test_u[idx]
    
v_min_1 = torch.min(u_field[0])
v_max_1 = torch.max(u_field[0])

v_min_2 = torch.min(u_field[int(T_out/ 2)])
v_max_2 = torch.max(u_field[int(T_out/ 2)])

v_min_3 = torch.min(u_field[-1])
v_max_3 = torch.max(u_field[-1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 3, 1)
pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t=' + str(0))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 2)
pcm = ax.imshow(u_field[int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t=' + str(int(T_out / 2)))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 3)
pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t=' + str(T_out))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx]

ax = fig.add_subplot(2, 3, 4)
pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 5)
pcm = ax.imshow(u_field[int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 6)
pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


plot_name = plot_loc + '/' + configuration['Field'] + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save_file(plot_name, 'output')

run.close()
# %%
