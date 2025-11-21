#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siren modelled over the GS Equations as an Implicit Neural Rep. 
Equation: Grad-Shafranov Equation from FreeGSNKE. Inputs: R, Z coordinates. Outputs: Magnetic Potential (Psi), conditioning(context): PF Coil locations.

"""

# %%
configuration = {"Case": 'Grad-Shafranov',
                 "Field": 'psi',
                 "Model": 'Siren', #Siren or MFN
                 "Epochs": 500,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "Layers": 5,                 
                 "Width": 64, 
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
from Neural_PDE.Models.INR_base import *
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
coords_input = torch.column_stack((RR.flatten(), ZZ.flatten()))
psi_output = psi.flatten(1,2).unsqueeze(-1)

#Normalisations. 
normalizer = MinMax_Normalizer
normalizer_RZ = normalizer(coords_input)
normalizer_psi = normalizer(psi_output)

coords_input = normalizer_RZ.encode(coords_input)
psi_output = normalizer_psi.encode(psi_output)

#Broadcasting
A_expanded = np.expand_dims(pf_loc, 1).repeat(len(coords_input), axis=1)  
# Expand and broadcast B
B_expanded = np.expand_dims(coords_input, 0).repeat(len(pf_loc), axis=0)  
# Concatenate along last axis
inputs = torch.tensor(np.concatenate([A_expanded, B_expanded], axis=2), dtype=torch.float32)  # Shape: [50, 1024, 14]
outputs = psi_output

#Test-Train Split
ins_train = inputs[:300]
outs_train = outputs[:300]
ins_test = inputs[300:]
outs_test = outputs[300:]

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################

def train_one_epoch_INR(model, ins_train, outs_train, ins_test, outs_test, loss_func, optimizer):
    model.train()
    train_loss = 0
    for idx in range(len(ins_train)):
        inputs = ins_train[idx].to(device)
        outputs = outs_train[idx].to(device)
        optimizer.zero_grad()
        im = model(inputs)[0]
        loss = loss_func(im, outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss =  train_loss / (len(ins_train))
    
    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for idx in range(len(ins_test)):
            inputs = ins_test[idx].to(device)
            outputs = outs_test[idx].to(device)
            im = model(inputs)[0]
            test_loss += loss_func(im, outputs)

        test_loss =  train_loss / (len(ins_test))

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

def validation_INR(model, u0, uu):
    with torch.no_grad():
        pred = torch.zeros(uu.shape)
        for idx in range(len(ins_test)):
            inputs = u0[idx].to(device)
            outputs = uu[idx].to(device)
            im = model(inputs)[0]
            pred[idx] = im
            
        # Performance Metrics
        MSE_error = (uu - pred).pow(2).mean()
        MAE_error = torch.abs(uu - pred).mean()

    return pred, MSE_error, MAE_error


if configuration['Model'] == 'Siren':
    model = Siren(in_features=configuration['Coords']+configuration['Context'], hidden_features=configuration['Width'], hidden_layers=configuration['Layers'], out_features=configuration['Variables'])
elif configuration['Model'] == 'MFN':
    model = FourierNet(in_size=configuration['Coords']+configuration['Context'], hidden_size=configuration['Width'], n_layers=configuration['Layers'], out_size=configuration['Variables'])
model.to(device)

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
    train_loss, test_loss = train_one_epoch_INR(model, ins_train, outs_train, ins_test, outs_test, loss_func, optimizer)
    
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
pred_set_encoded, mse, mae = validation_INR(model, ins_test, outs_test)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#Denormalising and reshaping the target and the predictions
pred_set = normalizer_psi.decode(pred_set_encoded.to(device)).cpu()
pred_set = pred_set.reshape(len(pred_set), len(R), len(Z))

test_u = normalizer_psi.decode(outs_test.to(device)).cpu()
test_u = test_u.reshape(len(test_u), len(R), len(Z))

# %% 
#Plotting performance

idx = 0
u_field = test_u[idx]
    
v_min = torch.min(u_field)
v_max = torch.max(u_field)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
pcm = ax.contourf(u_field, cmap=matplotlib.cm.coolwarm, vmin=v_min, vmax=v_max)
ax.title.set_text('Data')
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx]
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
