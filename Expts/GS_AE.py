#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siren modelled over the GS Equations as an Implicit Neural Rep. 
Equation: Grad-Shafranov Equation from FreeGSNKE. Inputs: R, Z coordinates. Outputs: Magnetic Potential (Psi), conditioning(context): PF Coil locations.

"""

# %%
configuration = {"Case": 'Grad-Shafranov',
                 "Field": 'psi',
                 "Model": 'Conditional AE', #Siren or MFN
                 "Epochs": 500,
                 "Batch Size": 50, 
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "Coords": 2, #Number of spatio-temporal coordinates - would form the number of inputs 
                 "Variables":1, #Number of variables being modelled - would form the number of outputs. 
                 "Context": 12,
                 "Loss Function": 'MSE',
                 "UQ": 'None', #None, Dropout
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'Siren', 'INR', 'FreeGSNKE', configuration['Model']], metadata=configuration)

# Saving the current run file and the git hash of the repo
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
from torch.utils.data import Dataset, DataLoader
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
from Neural_PDE.Models.INR import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Settung up locations. 
file_loc = os.getcwd()
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/FreeGSNKE'
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
import json
t1 = default_timer()
with open(data_loc + '/nunn_ixd_vignesh-2_8_0_sobol_512-icmldata.json', 'r') as file:
    json_data = json.load(file)

pf_loc = []
psi = []
for ii in range(len(json_data)):
    pf_loc.append(json_data[ii]['x'])
    psi.append(np.asarray(json_data[ii]['psi']))

R = torch.tensor(np.asarray(json_data[ii]['R']), dtype=torch.float32)
Z = torch.tensor(np.asarray(json_data[ii]['Z']), dtype=torch.float32)
pf_loc = torch.tensor(np.asarray(pf_loc), dtype=torch.float32)
psi = torch.tensor(np.asarray(psi), dtype=torch.float32)

# %% 
#Setting up the Data for training 
RR, ZZ = torch.meshgrid(R, Z)
coords_input = torch.stack((RR, ZZ))
psi_output = psi.unsqueeze(1)

# %% 
from torch.utils.data import TensorDataset
#Normalisations. 
normalizer = MinMax_Normalizer
normalizer_RZ = normalizer(coords_input)
normalizer_psi = normalizer(psi_output)

coords_input = normalizer_RZ.encode(coords_input)
psi_output = normalizer_psi.encode(psi_output)

#Test-Train Split
pf_train = pf_loc[:300]
psi_train = psi_output[:300]
pf_test = pf_loc[300:]
psi_test = psi_output[300:]

train_loader = DataLoader(TensorDataset(pf_train, psi_train), batch_size=configuration['Batch Size'], shuffle=True) 
test_loader = DataLoader(TensorDataset(pf_test, psi_test), batch_size=configuration['Batch Size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################

def train_one_epoch_INR(model, train_loader, test_loader, loss_func, optimizer):
    model.train()
    train_loss = 0
    for xx, yy in train_loader:
        xx, yy = xx.to(device), yy.to(device)
        coords = coords_input.repeat(xx.shape[0], 1, 1, 1).to(device)
        optimizer.zero_grad()
        im = model(coords, xx)
        loss = loss_func(im, yy)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    
    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            coords = coords_input.repeat(xx.shape[0], 1, 1, 1).to(device)
            optimizer.zero_grad()
            im = model(coords, xx)

            test_loss += loss.item()

        train_loss =  train_loss / len(train_loader)
        test_loss =  test_loss / len(test_loader)

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

def validation_INR(model, u0, uu):
    with torch.no_grad():
        inputs = u0.to(device)
        outputs = uu.to(device)
        coords = coords_input.repeat(inputs.shape[0], 1, 1, 1).to(device)
        print(inputs.shape, coords.shape)
        im = model(coords, inputs)
        pred = im
            
        # Performance Metrics
        MSE_error = (outputs - pred).pow(2).mean()
        MAE_error = torch.abs(outputs - pred).mean()

    return pred, MSE_error, MAE_error


# %% 

import torch.nn as nn
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        
        # Feature integration
        fc_size = 64 * 33 * 33
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024 + 12, fc_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(8, out_channels, kernel_size=1)
        )
        
    def forward(self, x, additional_features):
        # Encode
        encoded = self.encoder(x)
        
        # Feature integration
        batch_size = x.size(0)
        flattened = encoded.view(batch_size, -1)
        fc1_out = torch.relu(self.fc1(flattened))
        combined = torch.cat([fc1_out, additional_features], dim=1)
        fc2_out = torch.relu(self.fc2(combined))
        
        # Reshape back to feature maps
        feature_maps = fc2_out.view(batch_size, 64, 33, 33)
        
        # Decode
        decoded = self.decoder(feature_maps)
        return decoded
    
    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams

model = ConvAutoencoder(in_channels=2, out_channels=1).to(device)
run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

# %%
####################################
#Training Loop 
####################################
start_time = default_timer()
for ep in tqdm(range(epochs)): #Training Loop - Epochwise
    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch_INR(model, train_loader, test_loader, loss_func, optimizer)
    
    t2 = default_timer()

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,2)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time

# %%
#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')
# %%
#Testing 
pred_set_encoded, mse, mae = validation_INR(model, pf_test, psi_test)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#Denormalising and reshaping the target and the predictions
pred_set = normalizer_psi.decode(pred_set_encoded.to(device)).cpu()
test_u = normalizer_psi.decode(psi_test.to(device)).cpu()

# %% 
#Plotting performance

idx = 0
u_field = test_u[idx][0]
    
v_min = torch.min(u_field)
v_max = torch.max(u_field)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
pcm = ax.contourf(u_field, cmap=matplotlib.cm.coolwarm, vmin=v_min, vmax=v_max)
ax.title.set_text('Data')
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx][0]
ax = fig.add_subplot(1, 2, 2)
pcm = ax.contourf(u_field, cmap=matplotlib.cm.coolwarm,  vmin=v_min, vmax=v_max)
ax.title.set_text('Prediction')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

plot_name = plot_loc + '/' + configuration['Field'] + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save_file(plot_name, 'output')

run.close()
# %%
