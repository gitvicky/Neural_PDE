"""
Created on 12 Dec 2023

Obtaining the Physics Residuals as a measure of UQ on DeepOnet surrogate for 1D Advection Equation .

Equation: 
    U_t + v U_x = 0

"""

#%%
#Training Configuration - used as the config file for simvue.
configuration = {"Case": 'Burgers',
                 "Field": 'u',
                 "Model": 'DeepOnet',
                 "Epochs": 5000,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 1000,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Min-Max',
                 "Layers": 4,
                 "Width": 256, 
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
                 }

import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'Burgers' 'DeepONet', 'Tests'], metadata=configuration)

# Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

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
from Neural_PDE.Models.DeepONet import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %% 
t1 = default_timer()

# Burgers - My sim 
# data =  np.load(data_loc + '/Burgers_1d.npz')
# u_sol  = data['u']
# x = data['x']
# dt = data['dt']
# t_end = 1.25
# x = torch.tensor(x, dtype=torch.float32)
# t = torch.linspace(0, t_end, 50)
# t_end = 1.25
# x = torch.tensor(x, dtype=torch.float32)
# t = torch.linspace(0, t_end, 50)
# X,T = torch.meshgrid(x,t)
# X_loc = torch.column_stack((X.flatten(), T.flatten()))
# initial_locations = np.arange(0, len(x))
# u_sol = torch.tensor(u_sol, dtype=torch.float32)


#Burgers - Zongyis datasets 
from scipy.io import loadmat 
data = loadmat(data_loc + '/burgers_data_R10.mat')
a_sol = data['a']
u_sol = data['u']
x = torch.tensor(np.linspace(0, 1, 8192), dtype=torch.float32)
t = torch.tensor(np.linspace(0, 1 , 2), dtype=torch.float32)
X = x.unsqueeze(-1)
XY_loc = x.unsqueeze(-1)
initial_locations = np.arange(0, 8192)

a_sol = torch.tensor(u_sol, dtype=torch.float32).unsqueeze(1)
u_sol = torch.tensor(u_sol, dtype=torch.float32).unsqueeze(1)
u_sol = torch.concat((a_sol, u_sol), axis=1)

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

normalizer = normalizer(u_sol)

# #Saving Normalisation 
# saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '_' + 'norms.npz'

# np.savez(saved_normalisations, 
#         a= normalizer.a.numpy(), b = normalizer.b.numpy()
# )

# run.save_file(saved_normalisations, 'output')


# %% 
#Setting up the Datasets nad Loaders. 
dataset = DON_Dataset(normalizer.encode(u_sol), initial_locations)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=configuration['Batch Size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=configuration['Batch Size'])

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################
model = DeepONet(in_branch=len(x),
        width_branch=configuration['Width'],
        layers_branch=configuration['Layers'], 
        out_branch=configuration['Width'],
        in_trunk=2,
        width_trunk=configuration['Width'],
        layers_trunk=configuration['Layers'], 
        out_trunk=configuration['Width'])

model.to(device)

# run.update_metadata({'Number of Params': int(model.count_params())})
# print("Number of model params : " + str(model.count_params()))
model.load_state_dict(torch.load(model_loc + '/DeepOnet_Burgers_complete-dynamic.pth', map_location=device))

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
        for tt in range(1, len(t)):
            tr = torch.FloatTensor(torch.column_stack((XY_loc, torch.ones(X.flatten().shape)*t[tt]))).to(device)
            tr = tr.unsqueeze(0).repeat(br.shape[0], 1, 1)
            u_t = uu[:,tt].flatten(start_dim=1, end_dim=-1).to(device)
            
            optimizer.zero_grad()
            im = model(br, tr)
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
            for tt in range(1, len(t)):
                tr = torch.FloatTensor(torch.column_stack((XY_loc, torch.ones(X.flatten().shape)*t[tt]))).to(device)
                tr = tr.unsqueeze(0).repeat(br.shape[0], 1, 1)
                u_t = uu[:,tt].flatten(start_dim=1, end_dim=-1).to(device)
                im = model(br, tr)
                test_loss += loss_func(im, u_t).item()
        
        test_loss =  test_loss / (len(test_loader)*test_loader.batch_size)

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.


def validation_don(model, u0, uu):
    with torch.no_grad():
        pred = torch.zeros(uu.shape)
        br = u0.unsqueeze(1).to(device)
        for tt in range(1, len(t)):
            tr = torch.FloatTensor(torch.column_stack((XY_loc, torch.ones(X.flatten().shape)*t[tt]))).to(device)
            tr = tr.unsqueeze(0).repeat(br.shape[0], 1, 1)
            u_t = uu[:,tt].flatten(start_dim=1, end_dim=-1).to(device)
            im = model(br, tr)
            pred[:,tt] = im
            
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

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss, 5)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time

#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')
# %%
#Testing 
test_loader = DataLoader(dataset, batch_size=len(dataset))
test_u0, test_u_encoded = next(iter(test_loader))
pred_set_encoded, mse, mae = validation_don(model, test_u0, test_u_encoded)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

# %% 
#Denormalising and reshaping the target and the predictions
pred_set = normalizer.decode(pred_set_encoded.to(device)).cpu()
pred_set = pred_set.numpy().reshape(len(pred_set), len(t), len(x))

test_u = normalizer.decode(test_u_encoded.to(device)).cpu()
test_u = test_u.numpy().reshape(len(test_u), len(t), len(x))

# %%
#Plotting the surrogate performance against that of the test data. 

idx = np.random.randint(0, len(pred_set)) 
idx = 0
x_range = x

u_field_actual = test_u[idx]
u_field_pred = pred_set[idx]

# v_min = np.min(u_field_actual)
# v_max = np.min(u_field_actual)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,3,1)
pcm = ax.plot(x_range, u_field_actual[0], color='green', label='Actual')
pcm = ax.plot(x_range, u_field_pred[0], color='firebrick', label='Prediction')
# ax.set_ylim([v_min, v_max])
ax.title.set_text('t='+ str(0))

# ax = fig.add_subplot(1,3,2)
# pcm = ax.plot(x_range, u_field_actual[10], color='green', label='Actual')
# pcm = ax.plot(x_range, u_field_pred[10], color='firebrick', label='Prediction')
# # ax.set_ylim([v_min, v_max])
# ax.title.set_text('t='+ str(int((10))))
# ax.axes.yaxis.set_ticks([])

ax = fig.add_subplot(1,3,3)
pcm = ax.plot(x_range, u_field_actual[-1], color='green', label='Actual')
pcm = ax.plot(x_range, u_field_pred[-1], color='firebrick', label='Prediction')
ax.title.set_text('t='+str(1))
# ax.set_ylim([v_min, v_max])
ax.axes.yaxis.set_ticks([])
ax.legend()

plot_name = plot_loc + '/' + configuration['Field'] + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save_file(plot_name, 'output')

run.close()

# %%
